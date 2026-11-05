import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ann_lms_predictor import (ANNLMSPredictor, evaluate_learning_data, prediction_metrics,
                               temperature_features)


def learning_frame(count=20):
    rows = []
    for t in range(count):
        row = {'time': t, 'target_time': t + 1, 'horizon_seconds': .01,
               'core': 0, 'target_temperature': 50 + (t + 1) * .1}
        for slot in range(9):
            row.update({f'slot_{slot}_mask': int(slot == 0),
                        f'slot_{slot}_temperature': 50 + t * .1 if slot == 0 else 0,
                        f'slot_{slot}_throttled': (t % 4) / 4 if slot == 0 else 0})
        rows.append(row)
    return pd.DataFrame(rows)


class PredictorTests(unittest.TestCase):
    def test_nine_node_window_order_and_boundary_padding(self):
        features = temperature_features(np.arange(9) + 40, np.arange(9) / 10, 9, 3)
        order = [4, 1, 2, 5, 8, 7, 6, 3, 0]
        np.testing.assert_allclose(features[4, :9], (np.array(order) + 40) / 100)
        np.testing.assert_allclose(features[4, 9:], np.array(order) / 10)
        self.assertEqual(features[0, 1], 0)  # no northern tile at top edge
        self.assertEqual(features[0, 10], 0)

    def test_lms_matches_equation_with_the_original_prediction_features(self):
        model = ANNLMSPredictor(1, mu=.1)
        x = model.features([50], [.25])
        old = model.weights.copy()
        prediction = model.predict(x)
        error = model.update(x, [52], prediction)
        np.testing.assert_allclose(error, [2])
        np.testing.assert_allclose(model.weights, old + .1 * .02 * x)
        self.assertEqual(model.updates, 1)

    def test_relu_and_persistence_initialization(self):
        model = ANNLMSPredictor(2)
        x = model.features([40, 60], [0, 1])
        np.testing.assert_allclose(model.predict(x), [40, 60])
        model.weights[:] = -1
        np.testing.assert_equal(model.predict(x), [0, 0])

    def test_offline_neuron_fit_on_known_linear_relation(self):
        rng = np.random.default_rng(123)
        x = np.zeros((100, 18))
        x[:, 0] = rng.uniform(.4, .8, 100)
        x[:, 9] = rng.uniform(0, 1, 100)
        target = 100 * (.9 * x[:, 0] + .03 * x[:, 9])
        model = ANNLMSPredictor(1).fit(x, target, np.zeros(100, dtype=int))
        self.assertAlmostEqual(model.weights[0, 0], .9, places=5)
        self.assertAlmostEqual(model.weights[0, 9], .03, places=5)

    def test_online_adaptation_reduces_error_after_a_shift(self):
        model = ANNLMSPredictor(1, mu=.2)
        x = model.features([50], [0])
        errors = []
        for _ in range(60):
            prediction = model.predict(x)
            errors.append(abs(prediction[0] - 55))
            model.update(x, [55], prediction)
        self.assertLess(errors[-1], errors[0] / 10)

    def test_save_load_and_invalid_weights(self):
        model = ANNLMSPredictor(2, mu=.02)
        model.update(model.features([50, 55], [0, .5]), [51, 54])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            model.save(path)
            restored = ANNLMSPredictor.load(path)
            self.assertEqual(restored.to_dict(), model.to_dict())
            invalid = model.to_dict()
            invalid['weights'] = [[float('nan')] * 18] * 2
            with self.assertRaises(ValueError):
                ANNLMSPredictor.from_dict(invalid)

    def test_future_targets_cannot_change_the_offline_model_or_first_test_prediction(self):
        frame = learning_frame()
        first, predictions, scores = evaluate_learning_data(frame, 1)
        changed = frame.copy()
        changed.loc[changed.time >= scores['split_time'], 'target_temperature'] += 10
        second, changed_predictions, _ = evaluate_learning_data(changed, 1)
        np.testing.assert_equal(first.weights, second.weights)
        self.assertEqual(predictions.ann_lms.iloc[0], changed_predictions.ann_lms.iloc[0])
        self.assertTrue((predictions.time >= scores['split_time']).all())
        self.assertEqual(first.updates, 0)  # returned checkpoint is offline-only

    def test_variable_final_horizon_is_excluded(self):
        frame = learning_frame()
        frame.loc[len(frame) - 1, 'horizon_seconds'] = .005
        _, predictions, _ = evaluate_learning_data(frame, 1)
        self.assertNotIn(19, set(predictions.time))

    def test_overlapping_forecasts_wait_until_the_label_timestamp(self):
        frame = learning_frame()
        frame['target_time'] = frame.time + 3
        frame['horizon_seconds'] = .03
        frame.loc[frame.time >= 12, 'target_temperature'] += 10
        _, predictions, _ = evaluate_learning_data(frame, 1)
        np.testing.assert_equal(predictions.ann_lms.iloc[:3].to_numpy(),
                                predictions.frozen_ann.iloc[:3].to_numpy())
        self.assertNotEqual(predictions.ann_lms.iloc[3], predictions.frozen_ann.iloc[3])

    def test_invalid_training_and_time_inputs(self):
        model = ANNLMSPredictor(1)
        for args in [dict(mu=0), dict(mu=float('nan')), dict(scale=0), dict(cores=True)]:
            with self.assertRaises(ValueError):
                ANNLMSPredictor(**({'cores': 1} | args))
        with self.assertRaises(ValueError):
            model.features([50], [2])
        with self.assertRaises(ValueError):
            model.predict(np.zeros((2, 18)))
        with self.assertRaises(ValueError):
            model.fit(np.zeros((2, 18)), [1, 2], np.array([0, .5]))
        frame = learning_frame()
        frame.loc[0, 'target_time'] = -1
        with self.assertRaises(ValueError):
            evaluate_learning_data(frame, 1)

    def test_metrics_use_temperature_units(self):
        metrics = prediction_metrics([50, 52], [51, 50])
        self.assertEqual(metrics['mae_celsius'], 1.5)
        self.assertEqual(metrics['max_error_celsius'], 2)
        with self.assertRaises(ValueError):
            prediction_metrics([], [])
