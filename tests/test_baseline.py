import json
from pathlib import Path
import random
import unittest

from dag_baseline import (DAG, assign_priorities, divide_subgraphs,
                          graham_bound, local_paths, simulate)


def example():
    data = json.loads((Path(__file__).resolve().parents[1] /
                       "examples/paper_12_nodes.json").read_text())
    return DAG({int(v): c for v, c in data["wcet"].items()}, data["edges"])


class BaselineTests(unittest.TestCase):
    def test_dynamic_priority_assignment_matches_full_path_enumeration(self):
        rng = random.Random(20261105)
        for trial in range(80):
            dag = DAG({v: (1 if trial % 3 == 0 else rng.randint(1, 8)) for v in range(10)},
                      [(u, v) for u in range(10) for v in range(u + 1, 10) if rng.random() < .25])
            _, blocks, tail = divide_subgraphs(dag)
            expected, priority = {}, len(dag.wcet)
            for block in blocks:
                for path in local_paths(dag, block):
                    for node in path:
                        if node not in expected:
                            expected[node] = priority
                            priority -= 1
            for node in dag.order:
                if node in tail:
                    expected[node] = priority
                    priority -= 1
            self.assertEqual(assign_priorities(dag, blocks, tail), expected)

    def test_paper_graph_metrics(self):
        dag = example()
        self.assertEqual(dag.workload(), 27)
        self.assertEqual(dag.critical_path(), ([1, 4, 6, 7, 9, 11, 12], 14))
        self.assertEqual(dag.pred[7], {3, 6})
        self.assertEqual(dag.ancestors[7], {1, 3, 4, 6})
        self.assertEqual(graham_bound(dag, 2), 20.5)
        self.assertEqual([v for v in dag.order if not dag.succ[v]], [12])

    def test_paper_blocks(self):
        joins, blocks, tail = divide_subgraphs(example())
        self.assertEqual(joins, [7, 9, 12])
        self.assertEqual([set(b) for b in blocks],
                         [{1, 3, 4, 6}, {2, 5, 7}, {8, 9, 10, 11}])
        self.assertEqual(tail, [12])

    def test_paper_local_priorities(self):
        dag = example()
        _, blocks, tail = divide_subgraphs(dag)
        self.assertEqual(local_paths(dag, blocks[1]), [(2, 5), (7,)])
        p = assign_priorities(dag, blocks, tail)
        self.assertGreater(p[5], p[7])
        self.assertEqual(len(set(p.values())), 12)
        for earlier, later in zip(blocks, blocks[1:] + [tail]):
            self.assertGreater(min(p[v] for v in earlier), max(p[v] for v in later))

    def test_paper_schedule_and_no_block_barrier(self):
        dag = example()
        _, blocks, tail = divide_subgraphs(dag)
        events = simulate(dag, assign_priorities(dag, blocks, tail), 2)
        by_node = {e["node"]: e for e in events}
        self.assertEqual(max(e["finish"] for e in events), 15)
        self.assertLess(by_node[2]["start"], max(by_node[v]["finish"] for v in blocks[0]))

    def test_chain_and_single_node(self):
        for dag in [DAG({1: 2, 2: 3}, [(1, 2)]), DAG({1: 2}, [])]:
            _, blocks, tail = divide_subgraphs(dag)
            events = simulate(dag, assign_priorities(dag, blocks, tail), 4)
            self.assertEqual(max(e["finish"] for e in events), dag.workload())

    def test_shared_local_node_assigned_once(self):
        dag = DAG({1: 1, 2: 2, 3: 1, 4: 1}, [(1, 2), (1, 3), (2, 4), (3, 4)])
        p = assign_priorities(dag, [[1, 2, 3, 4]], [])
        self.assertEqual(len(set(p.values())), 4)
        self.assertGreater(p[2], p[3])
        with self.assertRaisesRegex(ValueError, "limit"):
            local_paths(dag, dag.order, max_paths=1)

    def test_simultaneous_completions_are_batched(self):
        dag = DAG({1: 1, 2: 1, 3: 1, 4: 1}, [(1, 3), (2, 4)])
        events = simulate(dag, {1: 4, 2: 3, 3: 1, 4: 2}, 2)
        self.assertEqual([e["node"] for e in events], [1, 2, 4, 3])

    def test_invalid_inputs(self):
        for wcet, edges in [({}, []), ({1: -1}, []), ({1: float("nan")}, []),
                            ({True: 1}, []), ({1: True}, []), ({1: 1e308, 2: 1e308}, []),
                            ({1: 1}, [(1, 2)]), ({1: 1, 2: 1}, [(1, 2), (2, 1)])]:
            with self.assertRaises(ValueError):
                DAG(wcet, edges)
        for cores in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                graham_bound(example(), cores)
        with self.assertRaises(ValueError):
            simulate(example(), {}, 2)

    def test_seeded_random_dags_schedule_invariants(self):
        # Validate scheduler independently of the paper partition on many topologies.
        rng = random.Random(2026)
        for _ in range(40):
            wcet = {v: rng.randint(1, 10) for v in range(1, 13)}
            edges = [(u, v) for u in wcet for v in wcet if u < v and rng.random() < .2]
            dag = DAG(wcet, edges)
            p = {v: rng.random() for v in wcet}
            for cores in [1, 2, 4, 16]:
                events = simulate(dag, p, cores)
                by_node = {e["node"]: e for e in events}
                self.assertEqual(set(by_node), set(wcet))
                for u, v in edges:
                    self.assertLessEqual(by_node[u]["finish"], by_node[v]["start"])
                for core in range(cores):
                    rows = [e for e in events if e["core"] == core]
                    for a, b in zip(rows, rows[1:]):
                        self.assertLessEqual(a["finish"], b["start"])
                for e in events:
                    self.assertEqual(e["finish"] - e["start"], wcet[e["node"]])
                finish = max(e["finish"] for e in events)
                self.assertLessEqual(finish, graham_bound(dag, cores) + 1e-9)
                # At every event time, no core is idle if an unstarted node is ready.
                for t in {e[k] for e in events for k in ["start", "finish"]}:
                    active = sum(e["start"] <= t < e["finish"] for e in events)
                    waiting = [v for v in wcet if by_node[v]["start"] > t and
                               all(by_node[u]["finish"] <= t for u in dag.pred[v])]
                    self.assertTrue(active == cores or not waiting)


if __name__ == "__main__":
    unittest.main()
