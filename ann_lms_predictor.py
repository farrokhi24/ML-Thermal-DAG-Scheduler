"""Single-neuron ReLU temperature prediction with causal LMS adaptation.

Chen & Liao, ISCAS 2020, Section III, equations (3)--(5). Inputs are
nine temperatures and nine throttling ratios, not DAG-node properties.
Temperatures and targets are scaled by 100 for numerical conditioning.
An update uses the stored features that generated the preceding prediction.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_model import grid_topology

OFFSETS = [(0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1),
           (1, 0), (1, -1), (0, -1), (-1, -1)]


def temperature_features(temperatures, throttles, cores, columns=None, scale=100.0):
    positions, _, _ = grid_topology(cores, columns)
    temperatures = np.asarray(temperatures, dtype=float)
    throttles = np.asarray(throttles, dtype=float)
    if temperatures.shape != (cores,) or throttles.shape != (cores,):
        raise ValueError('One temperature and throttle ratio is required per core')
    if not np.isfinite(temperatures).all() or not np.isfinite(throttles).all():
        raise ValueError('Features must be finite')
    if np.any((throttles < 0) | (throttles > 1)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Require throttle ratios in [0,1] and positive temperature scale')
    at = {position: core for core, position in positions.items()}
    features = np.zeros((cores, 18))
    for core, (r, c) in positions.items():
        for slot, (dr, dc) in enumerate(OFFSETS):
            neighbor = at.get((r + dr, c + dc))
            if neighbor is not None:
                features[core, slot] = temperatures[neighbor] / scale
                features[core, slot + 9] = throttles[neighbor]
    return features


class ANNLMSPredictor:
    def __init__(self, cores, columns=None, mu=.01, scale=100.0, horizon_seconds=None):
        _, _, columns = grid_topology(cores, columns)
        if isinstance(mu, bool) or isinstance(scale, bool) or not math.isfinite(mu) or mu <= 0 or not math.isfinite(scale) or scale <= 0:
            raise ValueError('mu and temperature scale must be finite and positive')
        if horizon_seconds is not None and (not math.isfinite(horizon_seconds) or horizon_seconds <= 0):
            raise ValueError('Prediction horizon must be finite and positive')
        self.cores, self.columns, self.mu, self.scale = cores, columns, mu, scale
        self.horizon_seconds = horizon_seconds
        self.weights = np.zeros((cores, 18))
        self.weights[:, 0] = 1.0  # Stable persistence initialization, not fitted test data.
        self.updates = 0

    def features(self, temperatures, throttles):
        return temperature_features(temperatures, throttles, self.cores, self.columns, self.scale)

    def _inputs(self, features):
        inputs = np.asarray(features, dtype=float)
        if inputs.shape != (self.cores, 18) or not np.isfinite(inputs).all():
            raise ValueError('Expected a finite cores-by-18 feature matrix')
        return inputs

    def predict(self, features):
        inputs = self._inputs(features)
        output = np.maximum(0, np.einsum('ij,ij->i', self.weights, inputs)) * self.scale
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite ANN prediction')
        return output

    def update(self, previous_features, observed, prediction=None):
        """LMS: w <- w + mu * (observed - predicted)/scale * previous_x.

        Call only after the target sensor sample arrives. No future feature
        vector is substituted for the vector that generated this prediction.
        """
        inputs = self._inputs(previous_features)
        target = np.asarray(observed, dtype=float)
        if target.shape != (self.cores,) or not np.isfinite(target).all():
            raise ValueError('Expected one finite sensor reading per core')
        predicted = self.predict(inputs) if prediction is None else np.asarray(prediction, dtype=float)
        if predicted.shape != target.shape or not np.isfinite(predicted).all():
            raise ValueError('Invalid previous prediction')
        error = (target - predicted) / self.scale
        weights = self.weights + self.mu * error[:, None] * inputs
        if not np.isfinite(weights).all():
            raise ValueError('LMS diverged; reduce mu or rescale features')
        self.weights = weights
        self.updates += 1
        return target - predicted

    def fit(self, features, targets, core_ids, ridge=1e-6):
        """Offline initialization of the paper's single linear neuron per core.

        Least squares fits the multiply-accumulate neuron before online LMS;
        ridge regularizes towards persistence when features are collinear.
        """
        x, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
        ids = np.asarray(core_ids)
        if x.ndim != 2 or x.shape[1] != 18 or y.shape != (len(x),) or ids.shape != y.shape:
            raise ValueError('Invalid offline training array shapes')
        if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Training data must be nonempty and finite')
        if not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= self.cores)):
            raise ValueError('Invalid training core IDs')
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError('ridge must be finite and positive')
        weights = self.weights.copy()
        for core in range(self.cores):
            selected = ids == core
            if not selected.any():
                raise ValueError('Training data must include every core')
            a, b = x[selected], y[selected] / self.scale
            prior = np.zeros(18)
            prior[0] = 1
            weights[core] = np.linalg.solve(a.T @ a + ridge * np.eye(18), a.T @ b + ridge * prior)
        if not np.isfinite(weights).all():
            raise ValueError('Offline fit produced nonfinite weights')
        self.weights = weights
        return self

    def to_dict(self):
        return {'version': 1, 'architecture': 'single-neuron ReLU, 9 temperatures + 9 throttle ratios',
                'cores': self.cores, 'columns': self.columns, 'mu': self.mu,
                'scale': self.scale, 'horizon_seconds': self.horizon_seconds,
                'weights': self.weights.tolist(), 'online_updates': self.updates}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + '\n')

    @classmethod
    def from_dict(cls, data):
        if data.get('version') != 1:
            raise ValueError('Unsupported model format')
        model = cls(data['cores'], data['columns'], data['mu'], data['scale'], data.get('horizon_seconds'))
        weights = np.asarray(data['weights'], dtype=float)
        if weights.shape != model.weights.shape or not np.isfinite(weights).all():
            raise ValueError('Invalid saved weights')
        updates = data.get('online_updates', 0)
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError('Invalid update count')
        model.weights, model.updates = weights, updates
        return model

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def learning_arrays(frame, scale=100.0):
    ids = frame['core'].to_numpy(float)
    if not np.isfinite(ids).all() or np.any(ids != np.floor(ids)):
        raise ValueError('Learning core IDs must be integers')
    for i in range(9):
        mask = frame[f'slot_{i}_mask'].to_numpy(float)
        throttle = frame[f'slot_{i}_throttled'].to_numpy(float)
        if not np.isin(mask, [0, 1]).all() or not np.isfinite(throttle).all() or np.any((throttle < 0) | (throttle > 1)):
            raise ValueError('Invalid neighborhood masks or throttle ratios')
    features = np.column_stack([frame[f'slot_{i}_temperature'].to_numpy(float) / scale *
                                frame[f'slot_{i}_mask'].to_numpy(float) for i in range(9)] +
                               [frame[f'slot_{i}_throttled'].to_numpy(float) *
                                frame[f'slot_{i}_mask'].to_numpy(float) for i in range(9)])
    return features, frame['target_temperature'].to_numpy(float), ids.astype(int)


def prediction_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    if actual.size == 0 or actual.shape != predicted.shape or not (
            np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError('Metrics require matching nonempty finite arrays')
    errors = predicted - actual
    return {'samples': int(errors.size), 'mae_celsius': float(np.mean(np.abs(errors))),
            'rmse_celsius': float(np.sqrt(np.mean(errors ** 2))),
            'max_error_celsius': float(np.max(np.abs(errors)))}


def evaluate_learning_data(frame, cores, columns=None, train_fraction=.6, mu=.01):
    """Chronological holdout; frozen ANN, online ANN+LMS and persistence.

    Discard variable-horizon tail rows, then split at a timestamp. Training
    targets cannot cross into the test input interval. Update only after each
    held-out prediction; per-core neurons do not share online updates.
    """
    if not 0 < train_fraction < 1:
        raise ValueError('train_fraction must be in (0,1)')
    required = {'time', 'target_time', 'horizon_seconds', 'target_temperature', 'core'}
    if not required <= set(frame):
        raise ValueError('Missing learning columns')
    frame = frame.sort_values(['time', 'core']).copy()
    for name in ['time', 'target_time', 'horizon_seconds', 'target_temperature']:
        if not np.isfinite(frame[name].to_numpy(float)).all():
            raise ValueError('Learning timestamps and targets must be finite')
    if np.any(frame.target_time <= frame.time) or np.any(frame.horizon_seconds <= 0):
        raise ValueError('Targets must be strictly in the future')
    if frame.duplicated(['time', 'core']).any():
        raise ValueError('Duplicate time/core samples')
    horizon = float(frame['horizon_seconds'].mode().iloc[0]) if len(frame) else 0
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError('Learning horizon must be positive')
    frame = frame[np.isclose(frame.horizon_seconds, horizon, rtol=1e-8, atol=0)]
    times = np.sort(frame.time.unique())
    if len(times) < 4:
        raise ValueError('At least four fixed-horizon sample times are required')
    cutoff = times[max(1, min(len(times) - 1, int(len(times) * train_fraction)))]
    train = frame[(frame.time < cutoff) & (frame.target_time <= cutoff)]
    test = frame[frame.time >= cutoff]
    if train.empty or test.empty:
        raise ValueError('Chronological split is empty')
    model = ANNLMSPredictor(cores, columns, mu, horizon_seconds=horizon)
    model.fit(*learning_arrays(train, model.scale))
    frozen = ANNLMSPredictor.from_dict(model.to_dict())
    online = ANNLMSPredictor.from_dict(model.to_dict())
    records = []
    pending = []
    for time, group in test.groupby('time', sort=True):
        # Labels can arrive after several intervening feature timestamps.
        # Never update at prediction time merely because an offline CSV already
        # contains the future target value.
        arrived = [entry for entry in pending if entry[0] <= time]
        pending = [entry for entry in pending if entry[0] > time]
        for _, previous_x, previous_target, previous_prediction in sorted(arrived, key=lambda entry: entry[0]):
            online.update(previous_x, previous_target, previous_prediction)
        group = group.sort_values('core')
        if list(group.core) != list(range(cores)):
            raise ValueError('Each test timestamp must include every core exactly once')
        if group.target_time.nunique() != 1:
            raise ValueError('All cores in a timestamp must share one target time')
        x, target, _ = learning_arrays(group, model.scale)
        adaptive_prediction, fixed_prediction = online.predict(x), frozen.predict(x)
        persistence = group.slot_0_temperature.to_numpy(float)
        for core in range(cores):
            records.append({'time': float(time), 'core': core,
                            'target_time': float(group.iloc[core].target_time),
                            'actual': float(target[core]), 'ann_lms': float(adaptive_prediction[core]),
                            'frozen_ann': float(fixed_prediction[core]), 'persistence': float(persistence[core])})
        pending.append((float(group.target_time.iloc[0]), x.copy(), target.copy(), adaptive_prediction.copy()))
    predictions = pd.DataFrame(records)
    metrics = {name: prediction_metrics(predictions.actual, predictions[name])
               for name in ['ann_lms', 'frozen_ann', 'persistence']}
    return model, predictions, {'split_time': float(cutoff), 'train_rows': len(train),
                                'test_rows': len(test), 'horizon_seconds': horizon,
                                'online_updates_before_predictions': online.updates, 'metrics': metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning', type=Path, required=True)
    parser.add_argument('--cores', type=int, required=True)
    parser.add_argument('--columns', type=int)
    parser.add_argument('--mu', type=float, default=.01)
    parser.add_argument('--out', type=Path, default=Path('outputs/predictor'))
    args = parser.parse_args()
    try:
        model, predictions, scores = evaluate_learning_data(pd.read_csv(args.learning), args.cores,
                                                            args.columns, mu=args.mu)
        args.out.mkdir(parents=True, exist_ok=True)
        model.save(args.out / 'model.json')
        predictions.to_csv(args.out / 'predictions.csv', index=False)
        (args.out / 'metrics.json').write_text(json.dumps(scores, indent=2, allow_nan=False) + '\n')
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
