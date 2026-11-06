"""Paired thermal benchmarks on independent DAGs and a disjoint calibration set.

This evaluates the combined PROJECT, not a reproduction of NoC throughput
figures. All policies share DAGs, per-task power, initial plant and deadlines.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from algorithm_3 import certified_response_bound
from ann_lms_predictor import ANNLMSPredictor, prediction_metrics
from dag_baseline import DAG, assign_priorities, check_cores, divide_subgraphs, simulate
from dag_scheduler import DAGGenerator
from thermal_scheduler import export_thermal_run, simulate_thermal_dag, training_arrays

POLICIES = ('none', 'reactive', 'predictive', 'budget')
NUMERIC_METRICS = ('makespan', 'energy_joules', 'peak_sensor_temperature_celsius',
                   'peak_event_temperature_celsius', 'sampled_violation_core_seconds',
                   'sampled_excess_degree_core_seconds', 'throttled_core_seconds')


def prepare(dag):
    _, blocks, tail = divide_subgraphs(dag)
    return assign_priorities(dag, blocks, tail)


def calibrate(cores, settings, seeds, workload, parallelism, mu):
    xs, ys, ids = [], [], []
    for seed in seeds:
        wcet, edges = DAGGenerator(seed=seed, total_workload=workload,
                                  max_parallelism=parallelism).generate_dag()
        dag = DAG(wcet, edges)
        result = simulate_thermal_dag(dag, prepare(dag), cores, policy='random', seed=seed, **settings)
        x, y, core_ids = training_arrays(result, settings['dt'], settings['seconds_per_unit'])
        xs.append(x)
        ys.append(y)
        ids.append(core_ids)
    model = ANNLMSPredictor(cores, settings.get('columns'), mu=mu,
                           horizon_seconds=settings['dt'] * settings['seconds_per_unit'])
    model.fit(np.concatenate(xs), np.concatenate(ys), np.concatenate(ids))
    return model, sum(len(y) for y in ys)


def summarize(frame):
    """Means use successful runs; pair deltas require both baseline and treatment.

    Explicit failure/missing-pair counts prevent silent selection of easy DAGs.
    No significance claim is inferred from these descriptive statistics.
    """
    groups, paired = [], []
    for (cores, policy), rows in frame.groupby(['cores', 'policy'], sort=True):
        good = rows[rows.status == 'ok']
        item = {'cores': int(cores), 'policy': policy, 'requested': len(rows),
                'completed': len(good), 'failed': len(rows) - len(good)}
        for metric in NUMERIC_METRICS:
            values = pd.to_numeric(good[metric], errors='raise')
            item[metric] = {'mean': float(values.mean()) if len(values) else None,
                            'min': float(values.min()) if len(values) else None,
                            'max': float(values.max()) if len(values) else None}
        item['observed_deadline_met_fraction'] = float(good.deadline_met.mean()) if len(good) else None
        groups.append(item)
        if policy == 'none':
            continue
        baseline = frame[(frame.cores == cores) & (frame.policy == 'none')]
        both = rows[rows.status == 'ok'].merge(baseline[baseline.status == 'ok'], on='seed', suffixes=('', '_baseline'))
        delta = {'cores': int(cores), 'policy': policy, 'paired_samples': len(both),
                 'unpaired_samples': len(rows) - len(both)}
        for metric in NUMERIC_METRICS:
            delta[metric + '_mean_delta'] = float((both[metric] - both[metric + '_baseline']).mean()) if len(both) else None
        paired.append(delta)
    return {'by_policy': groups, 'paired_deltas_vs_none': paired}


def run_thermal_experiments(out, *, count=20, seed=2026, cores=(2, 4), workload=300.,
                            parallelism=8, seconds_per_unit=.05, dt=1., training_seed=9000,
                            training_graphs=3, limit=55., deadline_factor=1.25,
                            mu=None, export_traces=False, **parameters):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('count must be a positive integer')
    if isinstance(training_graphs, bool) or not isinstance(training_graphs, int) or training_graphs < 1:
        raise ValueError('training_graphs must be a positive integer')
    if not cores or len(cores) != len(set(cores)):
        raise ValueError('Provide distinct core counts')
    for c in cores:
        check_cores(c)
    for name, value in [('seconds_per_unit', seconds_per_unit), ('dt', dt),
                        ('deadline_factor', deadline_factor), ('workload', workload)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + ' must be positive and finite')
    train_seeds = list(range(training_seed, training_seed + training_graphs))
    test_seeds = list(range(seed, seed + count))
    if set(train_seeds) & set(test_seeds):
        raise ValueError('Calibration and test DAG seeds must be disjoint')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(seconds_per_unit=seconds_per_unit, dt=dt, limit=limit,
                    ambient=45., initial_temperature=45., resistance=.8, capacitance=3.,
                    coupling=.15, idle_power=2., power_min=10., power_max=20.,
                    hysteresis=2., margin=.5, levels=(1., .75, .5, .25, 0.))
    settings.update(parameters)
    effective_mu = dt * seconds_per_unit if mu is None else mu
    models, calibration = {}, []
    for c in cores:
        model, samples = calibrate(c, settings, train_seeds, workload, parallelism, effective_mu)
        models[c] = model
        model.save(out / f'model_cores_{c}.json')
        calibration.append({'cores': c, 'samples': samples, 'dag_seeds': train_seeds})
    records, validation = [], []
    for test_seed in test_seeds:
        data = DAGGenerator(seed=test_seed, total_workload=workload, max_parallelism=parallelism).to_dict()
        dag = DAG({int(v): c for v, c in data['wcet'].items()}, data['edges'])
        priorities = prepare(dag)
        canonical = json.dumps(data, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        power_seed = test_seed ^ 0x50A3
        inputs = out / 'inputs'
        inputs.mkdir(exist_ok=True)
        (inputs / f'dag_{test_seed}.json').write_text(canonical + '\n')
        for c in cores:
            certified, method = certified_response_bound(dag, c)
            # Explicit PROJECT deadline, set before evaluating any control policy.
            deadline = deadline_factor * certified
            first_powers = None
            for policy in POLICIES:
                row = {'seed': test_seed, 'power_seed': power_seed, 'cores': c, 'policy': policy,
                       'input_sha256': digest, 'status': 'ok', 'error': '', 'nodes': len(dag.wcet),
                       'deadline': deadline, 'deadline_met': None,
                       'certified_uncontrolled_bound': certified, 'uncontrolled_bound_method': method,
                       **{metric: None for metric in NUMERIC_METRICS}}
                try:
                    run = simulate_thermal_dag(dag, priorities, c, policy=policy, seed=power_seed,
                                               predictor=models[c], **settings)
                    if first_powers is None:
                        first_powers = run['node_powers']
                    elif first_powers != run['node_powers']:
                        raise RuntimeError('Policies used unequal task powers')
                    metrics = run['metrics']
                    row.update({k: metrics[k] for k in NUMERIC_METRICS})
                    row.update(deadline_met=metrics['makespan'] <= deadline,
                               progress_fallbacks=metrics['progress_fallbacks'],
                               prediction_mae=(metrics['prediction']['mae_celsius'] if metrics['prediction'] else None))
                    if policy == 'none':
                        expected = max(e['finish'] for e in simulate(dag, priorities, c))
                        if not math.isclose(expected, metrics['makespan'], rel_tol=1e-9, abs_tol=1e-9):
                            raise RuntimeError('Uncontrolled schedule no longer matches baseline')
                        # Paired predictor evaluation on EXACTLY the same plant trajectory:
                        # control is off for both, only LMS adaptation differs.
                        frozen = simulate_thermal_dag(dag, priorities, c, policy='none', seed=power_seed,
                                                      predictor=models[c], adapt=False, **settings)
                        x, y, _ = training_arrays(run, dt, seconds_per_unit)
                        if not np.allclose(run['trace'].filter(regex='^Temp_Core_'),
                                           frozen['trace'].filter(regex='^Temp_Core_')):
                            raise RuntimeError('Frozen/adaptive prediction trajectories differ')
                        validation.append({'seed': test_seed, 'cores': c,
                                           'online_ann_lms': metrics['prediction'],
                                           'frozen_ann': frozen['metrics']['prediction'],
                                           'persistence': prediction_metrics(y, x[:, 0] * models[c].scale)})
                    if export_traces:
                        export_thermal_run(run, out / f'seed_{test_seed}/cores_{c}/{policy}')
                except (ValueError, OverflowError) as error:
                    row.update(status='error', error=str(error), deadline_met=None)
                records.append(row)
            # Flush at case boundaries so long experiments remain inspectable.
            pd.DataFrame(records).to_csv(out / 'results.csv', index=False)
    frame = pd.DataFrame(records)
    report = summarize(frame)
    source_files = ['run_thermal_experiments.py', 'thermal_scheduler.py', 'thermal_model.py',
                    'ann_lms_predictor.py', 'dag_baseline.py', 'algorithm_3.py', 'dag_scheduler.py']
    source = Path(__file__).parent
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True, text=True)
    report.update(requested_runs=count * len(cores) * len(POLICIES),
                  failed_runs=int((frame.status != 'ok').sum()), calibration=calibration,
                  prediction_validation=validation,
                  settings={**settings, 'seed': seed, 'count': count, 'cores': list(cores),
                            'workload': workload, 'parallelism': parallelism, 'mu': effective_mu,
                            'training_seed': training_seed, 'training_graphs': training_graphs,
                            'deadline_factor': deadline_factor},
                  provenance={'python': platform.python_version(), 'git_head': git.stdout.strip(),
                              'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                                for name in source_files},
                              'packages': {name: importlib.metadata.version(name)
                                           for name in ['numpy', 'pandas', 'networkx']}},
                  limitations=['Synthetic RC parameters and project deadlines; not hardware measurements.',
                               'Each test DAG starts from the same initial temperature; not continuous multi-job traffic.',
                               'Exposure is a left-sensor-sample estimate; event peak is not a continuous maximum.',
                               'Deadline flags describe observed executions, not proven thermal timing guarantees.',
                               'Policy differences include elapsed idle energy and global measured cooldown.',
                               'Algorithm 3 candidate is not used to certify any thermal deadline.',
                               'budget adapts one-step reference power-budget control; predictive is project search.',
                               'Means are descriptive, including failure counts; no statistical superiority claim.'])
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def plot_results(results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.read_csv(results)
    frame = frame[frame.status == 'ok']
    if frame.empty:
        raise ValueError('No successful runs to plot')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric, label in zip(axes.flat,
            ['makespan', 'peak_event_temperature_celsius', 'energy_joules', 'sampled_violation_core_seconds'],
            ['Makespan (schedule units)', 'Event-sampled peak (C)', 'Energy (J)', 'Sampled exposure (core s)']):
        for policy in POLICIES:
            rows = frame[frame.policy == policy].groupby('cores')[metric].mean()
            ax.plot(rows.index, rows.values, marker='o', label=policy)
        ax.set_xlabel('Cores')
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
    axes.flat[0].legend()
    fig.suptitle('Paired synthetic DAG experiments: policy means')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('outputs/thermal_experiments'))
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--cores', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--workload', type=float, default=300)
    parser.add_argument('--parallelism', type=int, default=8)
    parser.add_argument('--seconds-per-unit', type=float, required=True)
    parser.add_argument('--dt', type=float, default=1)
    parser.add_argument('--limit', type=float, default=55)
    parser.add_argument('--deadline-factor', type=float, default=1.25)
    parser.add_argument('--training-seed', type=int, default=9000)
    parser.add_argument('--training-graphs', type=int, default=3)
    parser.add_argument('--mu', type=float)
    parser.add_argument('--hysteresis', type=float, default=2)
    parser.add_argument('--margin', type=float, default=.5)
    parser.add_argument('--ambient', type=float, default=45.)
    parser.add_argument('--initial-temperature', type=float, default=45.)
    parser.add_argument('--resistance', type=float, default=.8)
    parser.add_argument('--capacitance', type=float, default=3.)
    parser.add_argument('--coupling', type=float, default=.15)
    parser.add_argument('--idle-power', type=float, default=2.)
    parser.add_argument('--power-min', type=float, default=10.)
    parser.add_argument('--power-max', type=float, default=20.)
    parser.add_argument('--levels', type=float, nargs='+', default=[1., .75, .5, .25, 0.])
    parser.add_argument('--export-traces', action='store_true')
    args = parser.parse_args()
    try:
        report = run_thermal_experiments(**vars(args))
        plot_results(args.out / 'results.csv', args.out / 'comparison.png')
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({'requested_runs': report['requested_runs'], 'failed_runs': report['failed_runs'],
                      'paired_deltas_vs_none': report['paired_deltas_vs_none']}, indent=2))
    if report['failed_runs']:
        parser.exit(1, 'Some runs failed; inspect results.csv and summary.json.\n')


if __name__ == '__main__':
    main()
