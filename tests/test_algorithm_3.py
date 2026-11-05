import itertools
import json
from pathlib import Path
import random
import unittest

from algorithm_3 import (analyze_algorithm_3, calculate_algorithm_3_bound, max_antichain_size,
                          certified_response_bound, analyze_timing)
from dag_baseline import DAG, assign_priorities, divide_subgraphs, local_paths, simulate

ROOT = Path(__file__).resolve().parents[1]


def prepare(dag):
    _, blocks, tail = divide_subgraphs(dag)
    return blocks, tail, [local_paths(dag, b) for b in blocks], assign_priorities(dag, blocks, tail)


def load(name):
    data = json.loads((ROOT / 'examples' / name).read_text())
    return DAG({int(v): c for v, c in data['wcet'].items()}, data['edges'])


class Algorithm3Tests(unittest.TestCase):
    def test_large_path_count_keeps_scheduling_and_certified_analysis_available(self):
        layers = [list(range(3 + i * 2, 5 + i * 2)) for i in range(14)]
        wcet = {v: 1 for v in range(1, 32)}
        wcet[2] = 1000
        edges = [(1, 2), (2, 31)] + [(1, v) for v in layers[0]]
        for previous, current in zip(layers, layers[1:]):
            edges.extend((u, v) for u in previous for v in current)
        edges.extend((v, 31) for v in layers[-1])
        dag = DAG(wcet, edges)
        _, blocks, tail = divide_subgraphs(dag)
        priorities = assign_priorities(dag, blocks, tail)
        self.assertEqual(len(priorities), len(wcet))
        events = simulate(dag, priorities, 2)
        observed = max(e['finish'] for e in events)
        analysis = analyze_timing(dag, blocks, tail, priorities, 2, observed_makespan=observed)
        self.assertEqual(analysis['status'], 'not_evaluated_path_limit')
        self.assertIsNone(analysis['bound'])
        self.assertGreaterEqual(analysis['certified_response_bound'], observed)

    def test_certified_special_cases_do_not_rely_on_the_paper_candidate(self):
        dag = DAG({1: 1, 2: 2, 3: 3, 4: 1}, [(1, 2), (1, 3), (2, 4), (3, 4)])
        self.assertEqual(certified_response_bound(dag, 1), (7, 'serial_workload'))
        self.assertEqual(certified_response_bound(dag, 2), (5, 'sufficient_cores_critical_path'))
        paper = load('paper_12_nodes.json')
        self.assertEqual(certified_response_bound(paper, 2), (20.5, 'graham'))
        self.assertEqual(certified_response_bound(paper, 12), (14, 'sufficient_cores_critical_path'))

    def test_paper_intermediate_results(self):
        dag = load('paper_12_nodes.json')
        result = analyze_algorithm_3(dag, *prepare(dag), 2, observed_makespan=15)
        self.assertEqual(result['bound'], 16.5)
        self.assertEqual([b['response_bound'] for b in result['blocks']], [8.5, 4, 3])
        self.assertEqual(result['tail_length'], 1)
        self.assertEqual(result['blocks'][1]['residual_before'][2], 0)
        self.assertEqual(result['blocks'][1]['residual_before'][5], 2)
        self.assertEqual(result['blocks'][2]['residual_before'][8], 0)
        self.assertEqual(result['blocks'][2]['residual_before'][10], 3)
        self.assertEqual(result['blocks'][0]['node_analysis'][3]['explicit_interference'], [2, 4, 6])
        self.assertEqual(result['model_safe_bound'], 20.5)
        self.assertEqual(result['status'], 'experimental_unverified')
        self.assertFalse(result['is_proven_upper_bound'])
        self.assertEqual(calculate_algorithm_3_bound(dag, *prepare(dag), 2), 16.5)

    def test_counterexample_is_exposed_not_silently_clamped(self):
        dag = load('algorithm_3_counterexample.json')
        args = prepare(dag)
        observed = max(e['finish'] for e in simulate(dag, args[-1], 2))
        result = analyze_algorithm_3(dag, *args, 2, observed_makespan=observed)
        self.assertEqual(observed, 18)
        self.assertEqual(result['bound'], 23.5)
        self.assertEqual(result['status'], 'experimental_unverified')
        self.assertFalse(result['sanity_issues'])
        self.assertGreaterEqual(result['model_safe_bound'], observed)
        # Regression for the source of the discrepancy: v6 is treated as
        # completed at B2 release, although its simulated finish is 14 > 10.
        self.assertEqual(result['blocks'][1]['residual_before'][6], 5)

    def test_chain_and_single_node(self):
        for dag in [DAG({1: 2}, []), DAG({1: 2, 2: 3.5}, [(1, 2)])]:
            for cores in [1, 2, 8]:
                self.assertEqual(analyze_algorithm_3(dag, *prepare(dag), cores)['bound'], dag.workload())

    def test_width_matches_independent_subset_oracle(self):
        rng = random.Random(311)
        for _ in range(40):
            dag = DAG({v: 1 for v in range(8)}, [(u, v) for u in range(8)
                      for v in range(u + 1, 8) if rng.random() < .25])
            subset = {v for v in range(8) if rng.random() < .8}
            costs = {1: {v: rng.choice([0, 1]) for v in dag.order}}
            active = sorted(v for v in subset if costs[1][v])
            expected = 0
            for size in range(len(active) + 1):
                for nodes in itertools.combinations(active, size):
                    if all(u not in dag.ancestors[v] and v not in dag.ancestors[u]
                           for u, v in itertools.combinations(nodes, 2)):
                        expected = max(expected, size)
            self.assertEqual(max_antichain_size(subset, 1, dag, costs), expected)

    def test_width_uses_paths_outside_subset(self):
        dag = DAG({1: 1, 2: 1, 3: 1}, [(1, 2), (2, 3)])
        self.assertEqual(max_antichain_size({1, 3}, 1, dag, {1: dag.wcet}), 1)

    def test_invalid_analysis_inputs(self):
        dag = load('paper_12_nodes.json')
        blocks, tail, paths, priorities = prepare(dag)
        for cores in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                analyze_algorithm_3(dag, blocks, tail, paths, priorities, cores)
        with self.assertRaisesRegex(ValueError, 'path list'):
            analyze_algorithm_3(dag, blocks, tail, paths[:-1], priorities, 2)
        with self.assertRaisesRegex(ValueError, 'complete'):
            analyze_algorithm_3(dag, blocks, tail, [paths[0][:-1]] + paths[1:], priorities, 2)
        with self.assertRaisesRegex(ValueError, 'Algorithm 2'):
            analyze_algorithm_3(dag, blocks, tail, paths, {v: 1 for v in dag.wcet}, 2)
        with self.assertRaisesRegex(ValueError, 'Algorithm 1'):
            analyze_algorithm_3(dag, blocks[::-1], tail, paths, priorities, 2)
        multiple = DAG({1: 1, 2: 1}, [])
        with self.assertRaisesRegex(ValueError, 'source'):
            analyze_algorithm_3(multiple, *prepare(multiple), 2)

    def test_fractional_subtraction_does_not_invent_started_nodes(self):
        # An ulp-sized reduction in (12) must not remove a node's start delay.
        from dag_scheduler import DAGGenerator
        wcet, edges = DAGGenerator(seed=43, max_parallelism=5).generate_dag()
        dag = DAG(wcet, edges)
        result = analyze_algorithm_3(dag, *prepare(dag), 1)
        self.assertAlmostEqual(result['bound'], dag.workload())
        self.assertFalse(result['blocks'][1]['node_analysis'][7]['residual_reduced'])
