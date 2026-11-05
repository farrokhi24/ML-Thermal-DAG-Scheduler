"""DAG execution coupled to thermal sensing, ANN+LMS and activity throttling.

A duty ratio scales both execution progress and dynamic power. A task retains
its core while throttled; no task migration or DVFS is assumed. Duty ratios
are continuous averages over a sensing interval, not a transistor-level model.
The 2020 prediction paper does not specify a full ratio-selection algorithm;
our discrete predictive policy and hysteresis are explicit configurable choices.
"""
import heapq
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd

from ann_lms_predictor import ANNLMSPredictor, prediction_metrics
from dag_baseline import check_cores
from thermal_model import RCNetwork, _positive, _write_learning_data


class ThrottleController:
    def __init__(self, cores, limit=55.0, hysteresis=2.0, margin=.5,
                 levels=(1.0, .75, .5, .25, 0.0)):
        check_cores(cores)
        _positive('hysteresis', hysteresis)
        _positive('margin', margin, zero=True)
        if not math.isfinite(limit) or not levels or levels[0] != 1 or levels[-1] != 0 or (
                sorted(set(levels), reverse=True) != list(levels)) or any(not 0 <= x <= 1 for x in levels):
            raise ValueError('Require finite limit and descending distinct duty levels from 1 to 0')
        self.cores, self.limit, self.hysteresis, self.margin = cores, limit, hysteresis, margin
        self.levels = tuple(levels)
        self.emergency_cooling = False
        self.progress_fallbacks = 0

    def decide(self, temperatures, occupied, mode, predictor=None):
        temperatures = np.asarray(temperatures, dtype=float)
        occupied = np.asarray(occupied, dtype=bool)
        if mode not in ['none', 'reactive', 'predictive']:
            raise ValueError('Unknown controller mode')
        if temperatures.shape != (self.cores,) or occupied.shape != (self.cores,) or not np.isfinite(temperatures).all():
            raise ValueError('Invalid sensor vector')
        duty = np.ones(self.cores)
        if mode == 'none':
            return duty
        # Global measured-temperature guard: cooling every core gives a finite,
        # model-bounded recovery time even with heat from adjacent busy cores.
        if temperatures.max() >= self.limit:
            self.emergency_cooling = True
        if temperatures.max() <= self.limit - self.hysteresis:
            self.emergency_cooling = False
        if self.emergency_cooling:
            duty[occupied] = 0
            return duty
        if mode == 'predictive':
            if predictor is None:
                raise ValueError('Predictive control requires a trained predictor')
            for core in range(self.cores):
                if not occupied[core]:
                    continue
                for level in self.levels:
                    duty[core] = level
                    predicted = predictor.predict(predictor.features(temperatures, 1 - duty))
                    if predicted[core] <= self.limit - self.margin:
                        break
                else:
                    duty[core] = 0
            # An inaccurate forecast must not stall a cool platform forever.
            # At least one occupied core makes progress outside emergency cooling.
            if occupied.any() and not np.any(duty[occupied] > 0):
                eligible = np.flatnonzero(occupied)
                core = min(eligible, key=lambda c: (temperatures[c], c))
                duty[core] = min(level for level in self.levels if level > 0)
                self.progress_fallbacks += 1
        return duty


def guarded_response_bound(workload, dt, seconds_per_unit, ambient, resistance,
                           capacitance, idle_power, maximum_power, initial_temperature,
                           limit, hysteresis, *, nodes, minimum_duty=.25):
    """Conservative bound for THIS global-cooling/minimum-progress policy.

    The RC maximum principle bounds all temperatures by Tmax. During global
    idle, max(T-Tidle) decays at least exp(-t/(R*C)); spatial coupling cannot
    increase the current maximum. Recovery adds at most one sensing interval.
    A no-completion active sensing interval makes >= minimum_duty progress.
    A completion can leave only paused tasks until the next sensing tick, so
    conservatively add nodes*dt for these partial intervals. Thus active time
    <= W/minimum_duty + nodes*dt, with at most ceil(active/dt)+1 cooling
    episodes (including a possible initial cooldown). This is not the
    DAG paper's bound and is not a guarantee that the threshold is never crossed.
    """
    for name, value in [('workload', workload), ('dt', dt), ('seconds_per_unit', seconds_per_unit),
                        ('resistance', resistance), ('capacitance', capacitance),
                        ('minimum_duty', minimum_duty)]:
        _positive(name, value)
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1:
        raise ValueError('nodes must be a positive integer')
    if minimum_duty > 1:
        raise ValueError('minimum_duty must not exceed 1')
    equilibrium = ambient + resistance * idle_power
    resume = limit - hysteresis
    if resume <= equilibrium:
        raise ValueError('Global cooldown cannot reach its recovery threshold')
    maximum = max(initial_temperature, ambient + resistance * maximum_power, equilibrium)
    cool_seconds = (0.0 if maximum <= resume else
                    resistance * capacitance * math.log((maximum - equilibrium) / (resume - equilibrium)))
    active = workload / minimum_duty + nodes * dt
    result = active + (math.ceil(active / dt) + 1) * (cool_seconds / seconds_per_unit + dt)
    if not math.isfinite(result):
        raise ValueError('Controlled response bound overflow')
    return result


def simulate_thermal_dag(dag, priorities, cores, *, seconds_per_unit, dt=1.0,
                         policy='predictive', predictor=None, adapt=True, seed=42,
                         ambient=45.0, resistance=.8, capacitance=3.0, coupling=.15,
                         columns=None, initial_temperature=None, idle_power=2.0,
                         power_min=10.0, power_max=20.0, node_powers=None,
                         limit=55.0, hysteresis=2.0, margin=.5,
                         max_time=None, max_steps=200000):
    check_cores(cores)
    for name, value in [('seconds_per_unit', seconds_per_unit), ('dt', dt)]:
        _positive(name, value)
    for name, value in [('idle_power', idle_power), ('power_min', power_min), ('power_max', power_max)]:
        _positive(name, value, zero=True)
    if power_min > power_max or power_min < idle_power:
        raise ValueError('Require idle_power <= power_min <= power_max')
    if policy not in ['none', 'reactive', 'predictive', 'random']:
        raise ValueError('Unknown thermal policy')
    if set(priorities) != set(dag.wcet) or not all(math.isfinite(p) for p in priorities.values()):
        raise ValueError('Every DAG node requires a finite priority')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    plant = RCNetwork(cores, ambient=ambient, resistance=resistance, capacitance=capacitance,
                      coupling=coupling, columns=columns, initial_temperature=initial_temperature)
    if policy in ['reactive', 'predictive'] and limit - hysteresis <= ambient + resistance * idle_power:
        raise ValueError('Cooling threshold is below the idle equilibrium; recovery is impossible')
    model = None if predictor is None else ANNLMSPredictor.from_dict(predictor.to_dict())
    if policy == 'predictive' and model is None:
        raise ValueError('Predictive control requires a trained predictor')
    if model is not None and (model.cores != cores or model.columns != plant.columns):
        raise ValueError('Predictor topology must match the physical core grid')
    if model is not None and model.horizon_seconds is not None and not math.isclose(
            model.horizon_seconds, dt * seconds_per_unit, rel_tol=1e-10, abs_tol=0):
        raise ValueError('Predictor horizon must match the physical sensing interval')
    controller = ThrottleController(cores, limit, hysteresis, margin)
    power_rng, action_rng = random.Random(seed), random.Random(seed ^ 0xC011)
    powers = ({v: power_rng.uniform(power_min, power_max) for v in sorted(dag.wcet)}
              if node_powers is None else {int(v): float(p) for v, p in node_powers.items()})
    if set(powers) != set(dag.wcet) or any(not math.isfinite(p) or p < idle_power for p in powers.values()):
        raise ValueError('Each node requires finite active power >= idle power')
    controlled_bound = (guarded_response_bound(
        dag.workload(), dt, seconds_per_unit, ambient, resistance, capacitance, idle_power,
        max(powers.values()), ambient if initial_temperature is None else initial_temperature,
        limit, hysteresis, nodes=len(dag.wcet)) if policy in ['reactive', 'predictive'] else None)
    max_time = (max(100.0, controlled_bound if controlled_bound is not None else dag.workload() * 100)
                if max_time is None else max_time)
    _positive('max_time', max_time)
    dependencies = {v: len(dag.pred[v]) for v in dag.order}
    ready = [(-priorities[v], v) for v in dag.order if not dependencies[v]]
    heapq.heapify(ready)
    running, completed, remaining = {}, set(), dict(dag.wcet)
    events, records, decisions, predictions = {}, [], [], []
    time, energy, throttled_core_seconds = 0.0, 0.0, 0.0
    event_peak = float(np.max(plant.temperatures))
    pending = None

    def dispatch():
        for core in range(cores):
            if core not in running and ready:
                _, node = heapq.heappop(ready)
                running[core] = node
                events[node] = {'node': node, 'core': core, 'start': time, 'finish': None,
                                'duration': None, 'wcet': dag.wcet[node], 'priority': priorities[node]}

    def power_vector(duty):
        return np.array([idle_power + duty[c] * (powers[running[c]] - idle_power)
                         if c in running else idle_power for c in range(cores)])

    def sensor_row(duty, predicted=None):
        temperature = plant.temperatures
        power = power_vector(duty)
        row = {'Time': time, 'Time_Seconds': time * seconds_per_unit}
        for c in range(cores):
            row.update({f'Temp_Core_{c}': float(temperature[c]),
                        f'Latent_Temp_Core_{c}': float(temperature[c]),
                        f'Power_Core_{c}': float(power[c]),
                        f'Active_Core_{c}': int(c in running and duty[c] > 0),
                        f'Throttled_Core_{c}': float(1 - duty[c]) if c in running else 0.0,
                        f'Predicted_Next_Temp_Core_{c}': float(temperature[c] if predicted is None else predicted[c]),
                        f'Prediction_Valid_Core_{c}': int(predicted is not None)})
        return row

    def receive_sample():
        nonlocal pending
        if pending is None:
            return
        x, predicted, origin_time, forecast_time = pending
        # A shortened terminal interval cannot label a full-horizon prediction.
        if math.isclose(time, forecast_time, rel_tol=1e-10, abs_tol=1e-12) and math.isclose(
                forecast_time - origin_time, dt, rel_tol=1e-10, abs_tol=1e-12):
            actual = plant.temperatures
            predictions.extend({'time': time, 'prediction_time': origin_time, 'target_time': forecast_time,
                                'core': c, 'actual': float(actual[c]),
                                'prediction': float(predicted[c])} for c in range(cores))
            if adapt:
                model.update(x, actual, predicted)
        pending = None

    for step in range(max_steps):
        dispatch()
        receive_sample()
        occupied = [c in running for c in range(cores)]
        if policy == 'random':
            duty = np.array([action_rng.choice([.25, .5, .75, 1.0]) for _ in range(cores)])
            duty[~np.array(occupied)] = 1
        else:
            duty = controller.decide(plant.temperatures, occupied, policy, model)
        predicted = None
        target_time = min((step + 1) * dt, max_time)
        if target_time <= time:
            raise ValueError('Thermal simulation exceeded max_time or lost time precision')
        if model is not None:
            x = model.features(plant.temperatures, [1 - duty[c] if occupied[c] else 0 for c in range(cores)])
            predicted = model.predict(x)
            pending = (x.copy(), predicted.copy(), time, target_time)
        records.append(sensor_row(duty, predicted))
        decisions.append({'time': time, **{f'duty_core_{c}': float(duty[c]) for c in range(cores)}})
        while time < target_time and len(completed) < len(dag.wcet):
            finishes = [time + remaining[v] / duty[c] for c, v in running.items() if duty[c] > 0]
            boundary = min([target_time] + finishes)
            interval = boundary - time
            if interval <= 0 or not math.isfinite(interval * seconds_per_unit):
                raise ValueError('Thermal simulation time precision failure')
            power = power_vector(duty)
            plant.advance(power, interval * seconds_per_unit)
            event_peak = max(event_peak, float(np.max(plant.temperatures)))
            energy += float(power.sum()) * interval * seconds_per_unit
            throttled_core_seconds += sum(1 - duty[c] for c in running) * interval * seconds_per_unit
            for c, v in running.items():
                remaining[v] = max(0.0, remaining[v] - interval * duty[c])
            time = boundary
            finished = [c for c, v in running.items() if remaining[v] <= 1e-11 * dag.wcet[v]]
            # Batch simultaneous completions before deciding which ready node runs.
            for c in sorted(finished):
                v = running.pop(c)
                completed.add(v)
                events[v]['finish'] = time
                events[v]['duration'] = time - events[v]['start']
                for successor in sorted(dag.succ[v]):
                    dependencies[successor] -= 1
                    if dependencies[successor] == 0:
                        heapq.heappush(ready, (-priorities[successor], successor))
            dispatch()
        if len(completed) == len(dag.wcet):
            receive_sample()
            records.append(sensor_row(np.ones(cores)))
            break
        if time >= max_time:
            raise ValueError('Thermal simulation exceeded max_time; controller may be unable to recover')
    else:
        raise ValueError('Thermal simulation exceeded max_steps')
    frame = pd.DataFrame(records)
    errors = (prediction_metrics([p['actual'] for p in predictions], [p['prediction'] for p in predictions])
              if predictions else None)
    temperatures = frame[[f'Temp_Core_{c}' for c in range(cores)]].to_numpy()
    metrics = {'policy': policy, 'makespan': time, 'energy_joules': energy,
               'peak_sensor_temperature_celsius': float(temperatures.max()),
               'peak_event_temperature_celsius': event_peak,
               'sensor_readings_above_limit': int(np.sum(temperatures > limit)),
               'throttled_core_seconds': throttled_core_seconds,
               'prediction': errors, 'lms_updates': 0 if model is None else model.updates,
               'temperature_limit_celsius': limit,
               'conservative_controlled_response_bound': controlled_bound,
               'progress_fallbacks': controller.progress_fallbacks,
               'hard_temperature_guarantee': False,
               'timing_bound_applicable': policy == 'none'}
    return {'schedule': pd.DataFrame(list(events.values())), 'trace': frame,
            'decisions': pd.DataFrame(decisions), 'predictions': pd.DataFrame(predictions),
            'metrics': metrics, 'node_powers': powers, 'model': model,
            'positions': plant.positions, 'columns': plant.columns}


def training_arrays(result, dt, seconds_per_unit, scale=100.0):
    """Use only full sensing intervals; terminal partial intervals are excluded."""
    frame = result['trace']
    cores = len(result['positions'])
    x, y, ids = [], [], []
    encoder = ANNLMSPredictor(cores, result['columns'], scale=scale)
    for i in range(len(frame) - 1):
        current, target = frame.iloc[i], frame.iloc[i + 1]
        if not math.isclose(target.Time - current.Time, dt, rel_tol=1e-9, abs_tol=1e-12):
            continue
        features = encoder.features([current[f'Temp_Core_{c}'] for c in range(cores)],
                                    [current[f'Throttled_Core_{c}'] for c in range(cores)])
        x.extend(features)
        y.extend(target[f'Temp_Core_{c}'] for c in range(cores))
        ids.extend(range(cores))
    if not x:
        raise ValueError('Not enough full sensing intervals for training')
    return np.asarray(x), np.asarray(y), np.asarray(ids, dtype=int)


def export_thermal_run(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result['schedule'].to_csv(directory / 'schedule.csv', index=False)
    result['trace'].to_csv(directory / 'thermal_trace.csv', index=False)
    result['decisions'].to_csv(directory / 'control.csv', index=False)
    if not result['predictions'].empty:
        result['predictions'].to_csv(directory / 'predictions.csv', index=False)
    _write_learning_data(result['trace'], directory / 'thermal_trace_learning.csv',
                         result['positions'], result['columns'])
    pd.DataFrame([{'node': v, 'power_watts': p} for v, p in result['node_powers'].items()]).to_csv(
        directory / 'node_power.csv', index=False)
