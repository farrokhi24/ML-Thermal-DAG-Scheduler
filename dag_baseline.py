"""Small, dependency-free baseline for Sections II and III of the DAG paper.

Node execution is non-preemptive. Analytical response-time Algorithm 3 lives
in algorithm_3.py; synthetic thermal traces live in thermal_model.py.
"""

from dataclasses import dataclass
import heapq
import math


@dataclass
class DAG:
    wcet: dict[int, float]
    edges: list[tuple[int, int]]

    def __post_init__(self):
        self.wcet = dict(self.wcet)
        self.edges = sorted(set(tuple(edge) for edge in self.edges))
        if not self.wcet:
            raise ValueError("A DAG must contain at least one node")
        if any(not isinstance(v, int) or isinstance(v, bool) for v in self.wcet):
            raise ValueError("Node IDs must be integers")
        if any(isinstance(c, bool) or not math.isfinite(c) or c <= 0 for c in self.wcet.values()):
            raise ValueError("WCETs must be finite and strictly positive")
        if not math.isfinite(sum(self.wcet.values())):
            raise ValueError("Total WCET must be finite; rescale execution times")
        self.pred = {v: set() for v in self.wcet}
        self.succ = {v: set() for v in self.wcet}
        for u, v in self.edges:
            if u not in self.wcet or v not in self.wcet:
                raise ValueError("Edge references an unknown node")
            self.pred[v].add(u)
            self.succ[u].add(v)
        remaining = {v: len(p) for v, p in self.pred.items()}
        ready = [v for v in self.wcet if not remaining[v]]
        heapq.heapify(ready)
        self.order = []
        while ready:
            v = heapq.heappop(ready)
            self.order.append(v)
            for w in sorted(self.succ[v]):
                remaining[w] -= 1
                if remaining[w] == 0:
                    heapq.heappush(ready, w)
        if len(self.order) != len(self.wcet):
            raise ValueError("Graph contains a cycle")
        self.ancestors = {}
        for v in self.order:
            self.ancestors[v] = set(self.pred[v])
            for u in self.pred[v]:
                self.ancestors[v].update(self.ancestors[u])

    def critical_path(self):
        """Longest WCET path; equal-length paths use lexicographic ID order."""
        lengths, paths = {}, {}
        for v in self.order:
            if not self.pred[v]:
                lengths[v], paths[v] = self.wcet[v], (v,)
            else:
                u = min(self.pred[v], key=lambda x: (-lengths[x], paths[x]))
                lengths[v] = lengths[u] + self.wcet[v]
                paths[v] = paths[u] + (v,)
        sink = min((v for v in self.order if not self.succ[v]),
                   key=lambda v: (-lengths[v], paths[v]))
        return list(paths[sink]), lengths[sink]

    def workload(self):
        return sum(self.wcet.values())


def check_cores(cores):
    if not isinstance(cores, int) or isinstance(cores, bool) or cores < 1:
        raise ValueError("cores must be a positive integer")


def graham_bound(dag, cores):
    check_cores(cores)
    _, length = dag.critical_path()
    return length + (dag.workload() - length) / cores


def divide_subgraphs(dag):
    """Algorithm 1: ancestors of critical-path joins minus earlier blocks."""
    path, _ = dag.critical_path()
    joins = [v for v in path if len(dag.pred[v]) > 1]
    assigned, blocks = set(), []
    for join in joins:
        block = [v for v in dag.order if v in dag.ancestors[join] - assigned]
        blocks.append(block)
        assigned.update(block)
    tail = [v for v in dag.order if v not in assigned]
    return joins, blocks, tail


def local_paths(dag, block, max_paths=10000):
    """Enumerate local source-to-sink paths for the small demo.

    Enumeration can be exponential; fail explicitly rather than silently
    truncating the paths and changing Algorithm 2's priorities.
    """
    members = set(block)
    paths = []
    stack = [(v,) for v in reversed(dag.order)
             if v in members and not (dag.pred[v] & members)]
    while stack:
        path = stack.pop()
        successors = sorted(dag.succ[path[-1]] & members)
        if not successors:
            paths.append(path)
            if len(paths) > max_paths:
                raise ValueError("Local path limit exceeded; use a scalable path representation")
        else:
            stack.extend(path + (v,) for v in reversed(successors))
    return sorted(paths, key=lambda path: (-sum(dag.wcet[v] for v in path), path))


def assign_priorities(dag, blocks, tail):
    """Algorithm 2: earlier blocks, then longer paths, then topological order.

    Larger numeric priority wins. Shared nodes receive priority only once.
    """
    groups = [set(block) for block in blocks] + [set(tail)]
    flat = [v for group in groups for v in group]
    if len(flat) != len(set(flat)) or set(flat) != set(dag.wcet):
        raise ValueError("Blocks and tail must partition all nodes")
    priorities, next_priority = {}, len(dag.wcet)
    for block in blocks:
        members = set(block)
        unassigned = set(block)
        # Enumerating every complete path is unnecessary: paths that introduce
        # no new node cannot affect priorities. Repeatedly find the longest
        # complete path containing an unassigned node (two-state DAG DP).
        # This gives the same order as the full sorted path list, in at most
        # |block| passes, without exponential path enumeration.
        while unassigned:
            best = {}
            for v in dag.order:
                if v not in members:
                    continue
                options = []
                predecessors = dag.pred[v] & members
                if not predecessors:
                    options.append((v,))
                for p in sorted(predecessors):
                    options.extend(path + (v,) for path in best[p].values())
                states = {}
                for path in options:
                    seen = any(node in unassigned for node in path)
                    current = states.get(seen)
                    if current is None or (-sum(dag.wcet[node] for node in path), path) < (
                            -sum(dag.wcet[node] for node in current), current):
                        states[seen] = path
                best[v] = states
            candidates = [best[v][True] for v in dag.order if v in members
                          and not (dag.succ[v] & members) and True in best[v]]
            path = min(candidates, key=lambda nodes: (-sum(dag.wcet[v] for v in nodes), nodes))
            for v in path:
                if v not in priorities:
                    priorities[v] = next_priority
                    next_priority -= 1
                    unassigned.remove(v)
    for v in dag.order:
        if v in set(tail):
            priorities[v] = next_priority
            next_priority -= 1
    return priorities


def simulate(dag, priorities, cores):
    """One simultaneous DAG release, WCET durations, identical cores.

    Dispatch only ready nodes, with no artificial barriers between blocks.
    Complete ALL simultaneous finishes before dispatching at that timestamp.
    """
    check_cores(cores)
    if set(priorities) != set(dag.wcet):
        raise ValueError("Exactly one priority is required for every node")
    if any(not math.isfinite(p) for p in priorities.values()):
        raise ValueError("Priorities must be finite numbers")
    remaining = {v: len(p) for v, p in dag.pred.items()}
    ready = [(-priorities[v], v) for v in dag.order if not remaining[v]]
    heapq.heapify(ready)
    free = list(range(cores))
    running, events = [], []
    time = 0
    while ready or running:
        while free and ready:
            _, v = heapq.heappop(ready)
            core = heapq.heappop(free)
            finish = time + dag.wcet[v]
            if not math.isfinite(finish) or finish <= time:
                raise ValueError("Time scale overflow or insufficient floating-point precision")
            events.append({"node": v, "core": core, "start": time,
                           "finish": finish, "duration": dag.wcet[v],
                           "priority": priorities[v]})
            heapq.heappush(running, (finish, core, v))
        if not running:
            break
        time = running[0][0]
        while running and running[0][0] == time:
            _, core, v = heapq.heappop(running)
            heapq.heappush(free, core)
            for w in sorted(dag.succ[v]):
                remaining[w] -= 1
                if remaining[w] == 0:
                    heapq.heappush(ready, (-priorities[w], w))
    if len(events) != len(dag.wcet):
        raise RuntimeError("Simulation did not execute every node")
    return events
