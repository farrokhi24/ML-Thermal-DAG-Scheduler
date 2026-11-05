"""End-to-end DAG scheduling, synthetic thermal data, ANN+LMS and throttling."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import numpy as np

from algorithm_3 import analyze_timing
from ann_lms_predictor import ANNLMSPredictor, prediction_metrics
from dag_baseline import DAG, assign_priorities, divide_subgraphs, simulate
from dag_scheduler import DAGGenerator
from thermal_scheduler import export_thermal_run, simulate_thermal_dag, training_arrays


def prepare(dag):
    _, blocks, tail = divide_subgraphs(dag)
    priorities = assign_priorities(dag, blocks, tail)
    return blocks, tail, priorities


def run_system(data, out, *, cores=2, seconds_per_unit, dt=1.0, policy='predictive',
               seed=42, training_seed=9000, training_graphs=3, training_workload=200.0,
               limit=55.0, coupling=.15, mu=None, columns=None, **plant_parameters):
    if policy not in ['none', 'reactive', 'predictive']:
        raise ValueError('Unsupported comparison policy')
    if isinstance(training_graphs, bool) or not isinstance(training_graphs, int) or training_graphs < 1:
        raise ValueError('training_graphs must be a positive integer')
    if not math.isfinite(seconds_per_unit) or seconds_per_unit <= 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError('seconds_per_unit and dt must be finite and positive')
    dag = DAG({int(v): c for v, c in data['wcet'].items()}, data['edges'])
    blocks, tail, priorities = prepare(dag)
    period, deadline = data.get('period'), data.get('deadline')
    if (period is None) != (deadline is None) or (period is not None and
            (isinstance(period, bool) or isinstance(deadline, bool) or not math.isfinite(period)
             or period <= 0 or deadline != period)):
        raise ValueError('Use a positive implicit deadline D=T or omit both period/deadline')
    reserved_seed = data.get('seed')
    training_seeds = list(range(training_seed, training_seed + training_graphs))
    validation_seed = training_seed + training_graphs
    if reserved_seed in training_seeds + [validation_seed]:
        raise ValueError('Evaluation DAG seed must differ from calibration/validation seeds')
    settings = dict(cores=cores, seconds_per_unit=seconds_per_unit, dt=dt,
                    limit=limit, coupling=coupling, columns=columns, ambient=45.0,
                    resistance=.8, capacitance=3.0, initial_temperature=None,
                    idle_power=2.0, power_min=10.0, power_max=20.0,
                    hysteresis=2.0, margin=.5)
    settings.update(plant_parameters)
    x_all, y_all, ids_all = [], [], []
    for sample_seed in training_seeds:
        wcet, edges = DAGGenerator(seed=sample_seed, total_workload=training_workload,
                                   max_parallelism=max(2, min(8, cores * 2))).generate_dag()
        calibration = DAG(wcet, edges)
        if calibration.wcet == dag.wcet and calibration.edges == dag.edges:
            raise ValueError('Calibration DAG must differ from evaluation DAG')
        _, _, p = prepare(calibration)
        run = simulate_thermal_dag(calibration, p, policy='random', seed=sample_seed, **settings)
        x, y, ids = training_arrays(run, dt, seconds_per_unit)
        x_all.append(x)
        y_all.append(y)
        ids_all.append(ids)
    # Paper choice mu=Delta_ts, in physical seconds and with scaled features.
    model = ANNLMSPredictor(cores, columns, mu=dt * seconds_per_unit if mu is None else mu,
                           horizon_seconds=dt * seconds_per_unit)
    model.fit(np.concatenate(x_all), np.concatenate(y_all), np.concatenate(ids_all))
    wcet, edges = DAGGenerator(seed=validation_seed, total_workload=training_workload,
                               max_parallelism=max(2, min(8, cores * 2))).generate_dag()
    validation_dag = DAG(wcet, edges)
    if validation_dag.wcet == dag.wcet and validation_dag.edges == dag.edges:
        raise ValueError('Prediction validation DAG must differ from evaluation DAG')
    _, _, validation_priorities = prepare(validation_dag)
    fixed = simulate_thermal_dag(validation_dag, validation_priorities, policy='none', predictor=model,
                                adapt=False, seed=validation_seed, **settings)
    adaptive = simulate_thermal_dag(validation_dag, validation_priorities, policy='none', predictor=model,
                                   adapt=True, seed=validation_seed, **settings)
    # Physical trajectory is identical: no controller is active during this comparison.
    v_x, v_y, _ = training_arrays(fixed, dt, seconds_per_unit)
    validation_scores = {
        'frozen_ann': fixed['metrics']['prediction'], 'ann_lms': adaptive['metrics']['prediction'],
        'persistence': prediction_metrics(v_y, v_x[:, 0] * model.scale),
        'validation_seed': validation_seed,
    }
    baseline = simulate_thermal_dag(dag, priorities, policy='none', predictor=model,
                                   seed=seed, **settings)
    controlled = simulate_thermal_dag(dag, priorities, policy=policy, predictor=model,
                                     seed=seed, **settings)
    standard = simulate(dag, priorities, cores)
    reference_makespan = max(e['finish'] for e in standard)
    if not math.isclose(reference_makespan, baseline['metrics']['makespan'], rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError('Uncontrolled thermal scheduler differs from the baseline scheduler')
    analysis = analyze_timing(dag, blocks, tail, priorities, cores,
                              observed_makespan=reference_makespan)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'input.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    model.save(out / 'model.json')
    export_thermal_run(baseline, out / 'baseline')
    export_thermal_run(controlled, out / 'controlled')
    export_thermal_run(fixed, out / 'validation_frozen')
    export_thermal_run(adaptive, out / 'validation_adaptive')
    if controlled['model'] is not None:
        controlled['model'].save(out / 'model_after_online_updates.json')
    metrics = {
        'baseline': baseline['metrics'], 'controlled': controlled['metrics'],
        'prediction_validation': validation_scores,
        'training': {'dag_seeds': training_seeds, 'samples': int(sum(len(y) for y in y_all)),
                     'policy': 'random duty in {.25,.5,.75,1}, with matching execution and power',
                     'validation_dag_seed': validation_seed, 'evaluation_dag_seed': reserved_seed,
                     'evaluation_power_seed': seed, 'mu': model.mu, 'temperature_scale': model.scale},
        'timing': {'uncontrolled_makespan': reference_makespan,
                   'algorithm_3_candidate': analysis['bound'], 'algorithm_3_status': analysis['status'],
                   'model_safe_bound': analysis['model_safe_bound'],
                   'certified_response_bound': analysis['certified_response_bound'],
                   'certified_response_method': analysis['certified_response_method'],
                   'controlled_analytical_bound': (analysis['certified_response_bound'] if policy == 'none' else
                                                    controlled['metrics']['conservative_controlled_response_bound']),
                   'controlled_bound_method': (analysis['certified_response_method'] if policy == 'none' else
                                               'global-cooling maximum principle and minimum-progress bound'),
                   'deadline': deadline,
                   'controlled_observed_deadline_met': (controlled['metrics']['makespan'] <= deadline
                                                        if deadline is not None else None)},
        'settings': {**settings, 'policy': policy, 'seed': seed},
        'provenance': {'python': platform.python_version(),
                       'packages': {name: importlib.metadata.version(name)
                                    for name in ['numpy', 'pandas', 'networkx']},
                       'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                         for name in ['run_system.py', 'thermal_scheduler.py', 'thermal_model.py',
                                                      'ann_lms_predictor.py', 'dag_baseline.py', 'algorithm_3.py',
                                                      'dag_scheduler.py']}},
        'assumptions': ['Synthetic RC grid and duty-averaged activity; not calibrated hardware',
                        'ANN: single-neuron ReLU, temperatures/throttles in a 3x3 neighborhood',
                        'Discrete ratio search, global cooldown and progress fallback are implementation choices, not specified by the 2020 paper',
                        'No migration or voltage/frequency scaling',
                        'Graham and the paper candidate apply only to the uncontrolled model',
                        'No hard continuous-time temperature guarantee is claimed'],
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).parent / 'examples/paper_12_nodes.json')
    parser.add_argument('--out', type=Path, default=Path('outputs/system'))
    parser.add_argument('--cores', type=int, default=2)
    parser.add_argument('--seconds-per-unit', type=float, required=True)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--policy', choices=['none', 'reactive', 'predictive'], default='predictive')
    parser.add_argument('--limit', type=float, default=55)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--training-seed', type=int, default=9000)
    parser.add_argument('--training-graphs', type=int, default=3)
    parser.add_argument('--training-workload', type=float, default=200)
    parser.add_argument('--coupling', type=float, default=.15)
    parser.add_argument('--columns', type=int)
    parser.add_argument('--mu', type=float)
    args = parser.parse_args()
    try:
        scores = run_system(json.loads(args.input.read_text()), args.out, cores=args.cores,
                            seconds_per_unit=args.seconds_per_unit, dt=args.dt, policy=args.policy,
                            seed=args.seed, training_seed=args.training_seed, training_graphs=args.training_graphs,
                            training_workload=args.training_workload, limit=args.limit,
                            coupling=args.coupling, columns=args.columns, mu=args.mu)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({'baseline': scores['baseline'], 'controlled': scores['controlled'],
                      'prediction_validation': scores['prediction_validation']}, indent=2))


if __name__ == '__main__':
    main()
