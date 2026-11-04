import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import networkx as nx

from dag_scheduler import DAGGenerator, REFERENCE_REVISION, uunifast_discard


class GeneratorTests(unittest.TestCase):
    def test_seeded_topology_and_workload_invariants(self):
        for seed in range(100):
            generator = DAGGenerator(seed=seed, total_workload=7.25, max_parallelism=8)
            wcet, edges = generator.generate_dag()
            graph = generator.G
            self.assertTrue(nx.is_directed_acyclic_graph(graph))
            sources = [v for v in graph if graph.in_degree(v) == 0]
            sinks = [v for v in graph if graph.out_degree(v) == 0]
            self.assertEqual(sources, [1])
            self.assertEqual(sinks, [max(wcet)])
            self.assertEqual(len(nx.descendants(graph, 1)), len(wcet) - 1)
            self.assertTrue(5 <= generator.depth <= 8)
            self.assertTrue(all(2 <= len(layer) <= 8 for layer in generator.layers[1:-1]))
            self.assertTrue(all(math.isfinite(w) and w > 0 for w in wcet.values()))
            self.assertAlmostEqual(math.fsum(wcet.values()), 7.25)
            self.assertTrue(all(graph.nodes[u]['layer'] < graph.nodes[v]['layer'] for u, v in edges))

    def test_against_pinned_upstream_topology_fixtures(self):
        fixtures = json.loads((Path(__file__).parent / 'fixtures/reference_topologies.json').read_text())
        self.assertEqual(fixtures['reference_revision'], REFERENCE_REVISION)
        for case in fixtures['cases']:
            _, edges = DAGGenerator(seed=case['seed'], min_width=case['min_width']).generate_dag()
            self.assertEqual([list(e) for e in edges], case['edges'])

    def test_repeat_generation_export_and_rng_isolation(self):
        random.seed(123)
        state = random.getstate()
        first = DAGGenerator(seed=2026)
        original = first.generate_dag()
        DAGGenerator(seed=77).generate_dag()
        self.assertEqual(original, first.generate_dag())
        self.assertEqual(original, DAGGenerator(seed=2026).generate_dag())
        self.assertEqual(state, random.getstate())
        with tempfile.TemporaryDirectory() as directory:
            path = first.export_json(Path(directory) / 'nested/dag.json')
            exported = json.loads(path.read_text())
            self.assertEqual(exported, first.to_dict())
            self.assertEqual(exported['seed'], 2026)
            self.assertIsNone(exported['deadline'])
            self.assertEqual({int(v): c for v, c in exported['wcet'].items()}, original[0])

    def test_zero_and_one_connection_probability_still_connected(self):
        for probability in [0, 1]:
            generator = DAGGenerator(connect_prob=probability)
            generator.generate_dag()
            graph = generator.G
            self.assertEqual(sum(graph.in_degree(v) == 0 for v in graph), 1)
            self.assertEqual(sum(graph.out_degree(v) == 0 for v in graph), 1)

    def test_implicit_deadline_and_metadata(self):
        data = DAGGenerator(period=200, time_unit='ms').to_dict()
        self.assertEqual(data['period'], data['deadline'])
        self.assertEqual(data['deadline'], 200)
        self.assertEqual(data['time_unit'], 'ms')
        self.assertEqual(data['generation']['rounding'], 'none')

    def test_uunifast_cap_and_fixed_sum(self):
        weights = uunifast_discard(10, 5, max_wcet=1, rng=random.Random(91))
        self.assertTrue(all(0 < w <= 1 for w in weights))
        self.assertAlmostEqual(math.fsum(weights), 5)
        self.assertEqual(uunifast_discard(3, 3, max_wcet=1), [1, 1, 1])
        self.assertEqual(uunifast_discard(1, .25), [.25])
        # W<n is valid with continuous weights, and cannot hang like integer rounding.
        self.assertAlmostEqual(sum(uunifast_discard(20, .5)), .5)

    def test_infeasible_or_pathological_requests_terminate(self):
        with self.assertRaises(ValueError):
            uunifast_discard(2, 3, max_wcet=1)
        class AlwaysZero:
            def random(self):
                return 0
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            uunifast_discard(2, 1, rng=AlwaysZero(), max_attempts=2)
        for kwargs in [dict(min_depth=2), dict(max_depth=4), dict(max_parallelism=1),
                       dict(connect_prob=2), dict(total_workload=float('nan')),
                       dict(period=-1), dict(seed=True), dict(max_wcet=0)]:
            with self.assertRaises(ValueError):
                DAGGenerator(**kwargs)
