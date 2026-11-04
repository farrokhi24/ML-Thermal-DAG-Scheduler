"""Reproducible single-DAG generator for Section V-A of the DAG paper.

Topology adapted from dag-gen-rnd v0.1, rnddag.py::gen_rnd (MIT):
Copyright (c) 2020 Xiaotian Dai. See third_party/dag-gen-rnd-LICENSE.
The paper's width 2..p overrides upstream's 1..p. The upstream default edge
probability .5 is an explicit assumption, not a published paper parameter.
"""
import argparse
import json
import math
from pathlib import Path
import random

import networkx as nx

REFERENCE_REVISION = "cd8df21a48047610234aaf990141ea128cdfb8bc"
REFERENCE_URL = f"https://github.com/automaticdai/dag-gen-rnd/blob/{REFERENCE_REVISION}/src/rnddag.py"


def uunifast_discard(n, total_workload, *, rng=None, max_wcet=None, max_attempts=10000):
    """Uniform continuous fixed-sum UUniFast with bounded rejection.

    Reject nonpositive components and components above max_wcet. With no
    supplied WCET cap, the cap is W (vacuous); the DAG paper does not specify
    a per-node cap. No integer rounding or random correction is performed.
    The original utilization formulation uses cap=1; these are WCET weights.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(total_workload) or total_workload <= 0:
        raise ValueError("total_workload must be finite and positive")
    cap = total_workload if max_wcet is None else max_wcet
    if not math.isfinite(cap) or cap <= 0 or cap * n < total_workload:
        raise ValueError("max_wcet cannot accommodate the requested workload")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if cap * n == total_workload:
        return [cap] * n  # Singleton feasible set, not a rejection loop.
    rng = random.Random() if rng is None else rng
    for _ in range(max_attempts):
        remaining, values = total_workload, []
        for i in range(1, n):
            next_sum = remaining * rng.random() ** (1.0 / (n - i))
            values.append(remaining - next_sum)
            remaining = next_sum
        values.append(total_workload - math.fsum(values))
        if all(0 < c <= cap and math.isfinite(c) for c in values):
            return values
    raise ValueError("UUniFast-discard exhausted max_attempts; relax the WCET cap")


class DAGGenerator:
    def __init__(self, seed=2026, total_workload=500, min_depth=5, max_depth=8,
                 max_parallelism=4, connect_prob=0.5, max_wcet=None,
                 period=None, time_unit="abstract time unit", min_width=2):
        for name, value in [("seed", seed), ("min_depth", min_depth),
                            ("max_depth", max_depth), ("max_parallelism", max_parallelism),
                            ("min_width", min_width)]:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 3 <= min_depth <= max_depth or not 1 <= min_width <= max_parallelism:
            raise ValueError("Require 3 <= min_depth <= max_depth and 1 <= min_width <= p")
        if not math.isfinite(total_workload) or total_workload <= 0:
            raise ValueError("total_workload must be finite and positive")
        if not math.isfinite(connect_prob) or not 0 <= connect_prob <= 1:
            raise ValueError("connect_prob must be in [0, 1]")
        if max_wcet is not None and (not math.isfinite(max_wcet) or max_wcet <= 0):
            raise ValueError("max_wcet must be finite and positive")
        if period is not None and (not math.isfinite(period) or period <= 0):
            raise ValueError("period must be finite and positive")
        if not isinstance(time_unit, str) or not time_unit.strip():
            raise ValueError("A time_unit label is required")
        self.seed, self.total_workload = seed, total_workload
        self.min_depth, self.max_depth = min_depth, max_depth
        self.max_parallelism, self.min_width = max_parallelism, min_width
        self.connect_prob, self.max_wcet = connect_prob, max_wcet
        self.period, self.time_unit = period, time_unit
        self.G, self.layers, self.depth = nx.DiGraph(), [], None
        self._data = None

    def generate_dag(self):
        """Regenerate the same sample on every call; never touch global RNG state."""
        rng = random.Random(self.seed)
        self.depth = rng.randint(self.min_depth, self.max_depth)
        graph = nx.DiGraph()
        graph.add_node(1, layer=0)
        layers, parents, childless, next_id = [[1]], [1], set(), 2
        for layer in range(1, self.depth - 1):
            current = list(range(next_id, next_id + rng.randint(self.min_width, self.max_parallelism)))
            next_id += len(current)
            graph.add_nodes_from((v, {"layer": layer}) for v in current)
            childless.update(parents)
            # v0.1: Bernoulli edges from the immediately preceding layer.
            for v in current:
                for u in parents:
                    if rng.random() < self.connect_prob:
                        graph.add_edge(u, v)
                        childless.discard(u)
            # v0.1: attach each orphan to the source (possibly skipping layers).
            for v in current:
                if graph.in_degree(v) == 0:
                    graph.add_edge(1, v)
            layers.append(current)
            parents = current
        sink = next_id
        graph.add_node(sink, layer=self.depth - 1)
        graph.add_edges_from((v, sink) for v in sorted(set(parents) | childless))
        layers.append([sink])
        weights = uunifast_discard(len(graph), self.total_workload, rng=rng,
                                  max_wcet=self.max_wcet)
        wcet = dict(zip(sorted(graph), weights))
        nx.set_node_attributes(graph, wcet, "wcet")
        self.G, self.layers = graph, layers
        self._data = {
            "name": f"Synthetic DAG seed={self.seed}", "time_unit": self.time_unit,
            "source": "DAG paper Section V-A; dag-gen-rnd v0.1 topology adaptation",
            "seed": self.seed, "workload": math.fsum(weights),
            "period": self.period, "deadline": self.period,
            "generation": {"reference_url": REFERENCE_URL,
                           "reference_revision": REFERENCE_REVISION,
                           "min_depth": self.min_depth, "max_depth": self.max_depth,
                           "generated_layer_count": self.depth, "min_width": self.min_width,
                           "max_parallelism": self.max_parallelism,
                           "connect_prob": self.connect_prob, "layers": layers,
                           "requested_workload": self.total_workload,
                           "wcet_algorithm": "continuous UUniFast-discard",
                           "max_wcet": self.max_wcet, "rounding": "none",
                           "assumptions": ["connect_prob=.5 is upstream's default, not specified by the paper",
                                           "Depth counts generated layers; source/sink repair edges can skip layers",
                                           "D=T if period is supplied; no numerical deadline distribution is assumed"]},
            "wcet": wcet, "edges": [list(e) for e in sorted(graph.edges)],
        }
        return dict(wcet), sorted(graph.edges)

    def to_dict(self):
        if self._data is None:
            self.generate_dag()
        # Isolate the cached sample from mutations by a caller.
        return json.loads(json.dumps(self._data))

    def export_json(self, filepath):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")
        return path

    def visualize(self):
        import matplotlib.pyplot as plt
        if self._data is None:
            self.generate_dag()
        nx.draw(self.G, nx.multipartite_layout(self.G, subset_key="layer"), with_labels=True)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workload", type=float, default=500)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--min-depth", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--connect-prob", type=float, default=.5)
    parser.add_argument("--max-wcet", type=float)
    parser.add_argument("--period", type=float)
    parser.add_argument("--time-unit", default="abstract time unit")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("outputs/generated"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    records = []
    try:
        for i in range(args.count):
            generator = DAGGenerator(seed=args.seed + i, total_workload=args.workload,
                                     min_depth=args.min_depth, max_depth=args.max_depth,
                                     max_parallelism=args.parallelism, connect_prob=args.connect_prob,
                                     max_wcet=args.max_wcet, period=args.period, time_unit=args.time_unit)
            path = generator.export_json(args.out / f"dag_{i:04d}.json")
            records.append({"file": path.name, "seed": args.seed + i})
    except ValueError as error:
        parser.error(str(error))
    (args.out / "manifest.json").write_text(json.dumps({"samples": records}, indent=2) + "\n")
    print(f"Saved {len(records)} reproducible DAG(s) in {args.out.resolve()}")


if __name__ == "__main__":
    main()
