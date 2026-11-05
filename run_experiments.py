import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from algorithm_3 import analyze_timing
from comparative_methods import he2019_priorities, zhao2020_priorities
from dag_baseline import DAG, assign_priorities, check_cores, divide_subgraphs, simulate
from dag_scheduler import DAGGenerator


def run_experiments(out, *, count=1000, seed=2026, cores=(2, 4), parallelism=8,
                    workload=500, thermal_count=0, seconds_per_unit=None, thermal_dt=1.0):
    out = Path(out)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('count must be a positive integer')
    if not cores or len(set(cores)) != len(cores):
        raise ValueError('Provide unique core counts')
    for core_count in cores:
        check_cores(core_count)
    if not isinstance(thermal_count, int) or not 0 <= thermal_count <= count:
        raise ValueError('thermal_count must be between zero and count')
    if thermal_count and seconds_per_unit is None:
        raise ValueError('Thermal generation requires explicit seconds_per_unit')
    records = []
    for index in range(count):
        sample_seed = seed + index
        generator = DAGGenerator(seed=sample_seed, total_workload=workload, max_parallelism=parallelism)
        source = generator.export_json(out / 'inputs' / f'dag_{index:04d}.json')
        data = generator.to_dict()
        dag = DAG({int(v): c for v, c in data['wcet'].items()}, data['edges'])
        for core_count in cores:
            record = {'sample': index, 'seed': sample_seed, 'cores': core_count,
                      'input': str(source.relative_to(out)), 'nodes': len(dag.wcet),
                      'status': '', 'error': '', 'candidate': '', 'observed': '',
                      'he2019': '', 'zhao2020': '',
                      'graham': '', 'candidate_over_graham': ''}
            case = out / f'sample_{index:04d}' / f'cores_{core_count}'
            try:
                _, blocks, tail = divide_subgraphs(dag)
                priorities = assign_priorities(dag, blocks, tail)
                events = simulate(dag, priorities, core_count)
                observed = max(e['finish'] for e in events)
                
                he2019_p = he2019_priorities(dag)
                he2019_events = simulate(dag, he2019_p, core_count)
                he2019_makespan = max(e['finish'] for e in he2019_events)
                
                zhao2020_p = zhao2020_priorities(dag)
                zhao2020_events = simulate(dag, zhao2020_p, core_count)
                zhao2020_makespan = max(e['finish'] for e in zhao2020_events)
                
                analysis = analyze_timing(dag, blocks, tail, priorities, core_count,
                                          observed_makespan=observed)
                                          
                record.update(status=analysis['status'], candidate=analysis['bound'],
                              observed=observed, 
                              he2019=he2019_makespan, zhao2020=zhao2020_makespan,
                              graham=analysis['model_safe_bound'],
                              candidate_over_graham=(analysis['bound'] / analysis['model_safe_bound']
                                                     if analysis['bound'] is not None else ''))
                                     
                if analysis['sanity_issues'] or index < thermal_count:
                    case.mkdir(parents=True, exist_ok=True)
                    (case / 'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False) + '\n')
                    with (case / 'schedule.csv').open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(events[0]))
                        writer.writeheader()
                        writer.writerows(events)
                if index < thermal_count:
                    from thermal_model import generate_thermal_trace
                    generate_thermal_trace(case / 'schedule.csv', case / 'thermal_trace.csv',
                                           cores=core_count, seed=sample_seed,
                                           seconds_per_unit=seconds_per_unit, dt=thermal_dt)
            except (ValueError, OverflowError) as error:
                record.update(status='error', error=str(error))
            records.append(record)
            
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        
    source_root = Path(__file__).resolve().parent
    provenance = {'python': platform.python_version(),
                  'packages': {name: importlib.metadata.version(name)
                               for name in ['numpy', 'pandas', 'networkx', 'matplotlib']},
                  'source_sha256': {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                                    for name in ['dag_baseline.py', 'algorithm_3.py', 'dag_scheduler.py',
                                                 'thermal_model.py', 'comparative_methods.py', 'run_experiments.py']
                                    if (source_root / name).exists()}}
    summary = {'requested_samples': count, 'requested_cases': count * len(cores),
               'statuses': dict(Counter(r['status'] for r in records)),
               'provenance': provenance,
               'settings': {'seed': seed, 'cores': list(cores), 'max_parallelism': parallelism,
                            'workload': workload, 'thermal_sample_count': thermal_count,
                            'seconds_per_unit': seconds_per_unit, 'thermal_dt': thermal_dt},
               'limitations': ['Algorithm 3 candidates are experimental, including cases with no detected violation',
                               'WCET simulation checks do not prove a worst-case bound',
                               'All candidates including counterexamples remain in results.csv']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('outputs/experiments'))
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--cores', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--parallelism', type=int, default=8)
    parser.add_argument('--workload', type=float, default=500)
    parser.add_argument('--thermal-count', type=int, default=0)
    parser.add_argument('--seconds-per-unit', type=float)
    parser.add_argument('--thermal-dt', type=float, default=1)
    args = parser.parse_args()
    try:
        result = run_experiments(args.out, count=args.count, seed=args.seed, cores=args.cores,
                                 parallelism=args.parallelism, workload=args.workload,
                                 thermal_count=args.thermal_count, seconds_per_unit=args.seconds_per_unit,
                                 thermal_dt=args.thermal_dt)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()