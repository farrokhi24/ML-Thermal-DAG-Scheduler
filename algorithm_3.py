"""Section IV, equations (6)--(18), of Zhang et al. (2026).

Analysis for the canonical Algorithms 1/2 partition and priorities, one
single-source/single-sink DAG, identical cores, non-preemptive nodes.
Detailed intermediate quantities are available through analyze_algorithm_3.
"""
import math

from dag_baseline import (assign_priorities, check_cores, divide_subgraphs,
                          graham_bound, local_paths)


def _width(nodes, dag):
    """Maximum antichain via Dilworth: |V| minus maximum bipartite matching.

    Edges use transitive reachability in the ORIGINAL DAG, including paths
    through nodes outside this subset. Unlike subset enumeration this is
    polynomial in the number of interference nodes.
    """
    nodes = sorted(nodes)
    adjacent = {u: [v for v in nodes if u in dag.ancestors[v]] for u in nodes}
    matched = {}

    def augment(u, seen):
        for v in adjacent[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in matched or augment(matched[v], seen):
                matched[v] = u
                return True
        return False

    return len(nodes) - sum(augment(u, set()) for u in nodes)


def max_antichain_size(node_set, k, dag, C):
    """Compatibility API; zero-residual nodes do not contribute parallelism."""
    return _width({v for v in node_set if C[k].get(v, 0) > 0}, dag)


def certified_response_bound(dag, cores):
    """Sound bound for the unthrottled work-conserving baseline model.

    One core executes exactly W work. If m >= DAG width, the union of running
    and ready nodes (an antichain) always fits, so precedence alone determines
    completion and the critical-path length is exact. Otherwise use Graham.
    No observed schedule or experimental Algorithm 3 value is used here.
    """
    check_cores(cores)
    if cores == 1:
        return dag.workload(), 'serial_workload'
    if cores >= _width(set(dag.wcet), dag):
        return dag.critical_path()[1], 'sufficient_cores_critical_path'
    return graham_bound(dag, cores), 'graham'



class AnalysisStateLimit(ValueError):
    """Explicit resource limit on distinct interference states, not path count."""


def compact_path_states(dag, block, hp, lp, residual, max_states=200000):
    """Exact DP for (13)--(17) without enumerating all complete paths.

    At the same ending node, paths with equal unions of higher/lower priority
    interference have identical future interference scores. Retain only their
    largest residual path workload. A representative path is kept for auditing.
    Distinct interference signatures can still grow exponentially; report that
    resource limit explicitly instead of approximating or dropping cases.
    """
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise ValueError('max_states must be a positive integer')
    members, states, counts = set(block), {}, {}
    retained = 0
    for v in dag.order:
        if v not in members:
            continue
        predecessors = sorted(dag.pred[v] & members)
        counts[v] = sum(counts[u] for u in predecessors) if predecessors else 1
        current = {}
        parents = ((signature, work, path) for u in predecessors
                   for signature, (work, path) in states[u].items()) if predecessors else [( (frozenset(), frozenset()), 0.0, ())]
        for (high, low), work, path in parents:
            signature = (high | frozenset(hp[v]), low | frozenset(lp[v]))
            candidate = (work + residual[v], path + (v,))
            previous = current.get(signature)
            if previous is None or (-candidate[0], candidate[1]) < (-previous[0], previous[1]):
                current[signature] = candidate
                if len(current) + retained > max_states:
                    raise AnalysisStateLimit(f'Interference-state budget exceeded ({max_states})')
        states[v] = current
        retained += len(current)
    sinks = [v for v in dag.order if v in members and not (dag.succ[v] & members)]
    terminal = [(signature, work, path) for v in sinks
                for signature, (work, path) in states[v].items()]
    return terminal, {'complete_path_count': sum(counts[v] for v in sinks),
                      'retained_states': retained, 'terminal_states': len(terminal)}


def analyze_algorithm_3(dag, blocks, tail, local_paths_list, priorities, cores,
                        *, observed_makespan=None, max_states=200000):
    """Return the reproduced candidate plus analysis and independent safety checks.

    Inputs are validated against the baseline's canonical Algorithms 1 and 2.
    Arbitrary partitions, tied priorities, multiple sources/sinks and truncated
    path sets are deliberately rejected rather than producing an unsafe bound.
    With no supplied lists, exact compact interference states replace enumeration.
    This is an experimental reproduction: the residual/interference recurrence
    has counterexamples. A separate Graham bound remains the safe model bound.
    """
    check_cores(cores)
    if sum(not dag.pred[v] for v in dag.order) != 1 or sum(
            not dag.succ[v] for v in dag.order) != 1:
        raise ValueError("Algorithm 3 requires exactly one source and one sink")
    _, expected_blocks, expected_tail = divide_subgraphs(dag)
    if ([set(b) for b in blocks] != [set(b) for b in expected_blocks]
            or list(tail) != expected_tail):
        raise ValueError("Use the canonical Algorithm 1 blocks and tail")
    if local_paths_list is not None:
        if len(local_paths_list) != len(blocks):
            raise ValueError("One complete path list is required for every block")
        for block, paths in zip(blocks, local_paths_list):
            if sorted(map(tuple, paths)) != sorted(local_paths(dag, block)):
                raise ValueError("Local path lists must be complete and unmodified")
    if priorities != assign_priorities(dag, blocks, tail):
        raise ValueError("Use the canonical Algorithm 2 priorities")

    descendants = {v: set() for v in dag.order}
    for v in reversed(dag.order):
        for u in dag.succ[v]:
            descendants[v].add(u)
            descendants[v].update(descendants[u])
    all_nodes = set(dag.wcet)
    concurrent = {v: all_nodes - dag.ancestors[v] - descendants[v] - {v}
                  for v in dag.order}  # (6)
    residual = dict(dag.wcet)
    details = []
    for index, block in enumerate(blocks):
        members = set(block)
        later = set(tail).union(*(set(b) for b in blocks[index + 1:]))
        parallel = set().union(*(concurrent[v] for v in block)) & later  # (7)
        scope = members | parallel
        response, explicit, hp, lp, node_details = {}, {}, {}, {}, {}
        width_cache = {}

        def interference(nodes):
            active = frozenset(v for v in nodes if residual[v] > 0)
            if active not in width_cache:
                width_cache[active] = _width(active, dag)
            width = width_cache[active]
            work = math.fsum(residual[v] for v in sorted(active))
            return width, work, 0.0 if width < cores else work / cores

        def select_lower(nodes):
            return set(sorted(nodes, key=lambda v: (-residual[v], v))[:cores - 1])

        for v in dag.order:
            if v not in scope:
                continue
            hp[v], lp[v], explicit[v] = set(), set(), set()
            # A reduced residual upper estimate does not prove that a node is
            # actually running in every possible schedule. Retain start delays:
            # applying the post-(11) started-node shortcut here is unsafe.
            reduced = (0 < residual[v] < dag.wcet[v] and not math.isclose(
                residual[v], dag.wcet[v], rel_tol=1e-12, abs_tol=0))
            predecessor = max((response.get(p, 0) for p in dag.pred[v]), default=0)
            if residual[v] == 0:
                response[v] = residual[v]
                width, work, delay = 0, 0.0, 0.0
            else:
                inherited = set().union(*(explicit.get(a, set())
                                          for a in dag.ancestors[v] 
                                          if node_details.get(a, {}).get('interference_delay', 0) > 0))
                
                candidates = {u for u in scope & concurrent[v] - inherited
                              if residual[u] > 0}
                hp[v] = {u for u in candidates if priorities[u] > priorities[v]}  # (8)
                lp[v] = candidates - hp[v]  # (9), priorities are unique
                explicit[v] = hp[v] | select_lower(lp[v])  # (10)
                width, work, delay = interference(explicit[v])
                response[v] = predecessor + residual[v] + delay  # (11)
            node_details[v] = {
                "residual": residual[v], "residual_reduced": reduced,
                "higher_priority_interference": sorted(hp[v]),
                "lower_priority_interference": sorted(lp[v]),
                "explicit_interference": sorted(explicit[v]),
                "interference_width": width, "interference_workload": work,
                "interference_delay": delay, "response_bound": response[v],
            }

        paths = []
        if local_paths_list is None:
            terminal, path_stats = compact_path_states(dag, block, hp, lp, residual, max_states)
        else:
            terminal = [( (set().union(*(hp[v] for v in path)),
                           set().union(*(lp[v] for v in path))),
                          math.fsum(residual[v] for v in path), tuple(path))
                         for path in local_paths_list[index]]
            path_stats = {'complete_path_count': len(terminal), 'terminal_states': len(terminal),
                          'retained_states': None}
        for (high, low), path_work, path in terminal:
            path_explicit = set(high) | select_lower(low)
            width, work, delay = interference(path_explicit)
            paths.append({"nodes": list(path), "residual_workload": path_work,
                          "explicit_interference": sorted(path_explicit),
                          "interference_width": width,
                          "interference_workload": work,
                          "interference_delay": delay,
                          "response_bound": path_work + delay})
        block_bound = max(p["response_bound"] for p in paths)  # (17)
        node_completion = max(response[v] for v in block)
        next_residual = dict(residual)
        for v in parallel:
            next_residual[v] = min(dag.wcet[v], max(0.0, response[v] - node_completion))  # (12)
            if math.isclose(next_residual[v], dag.wcet[v], rel_tol=1e-12, abs_tol=0):
                next_residual[v] = dag.wcet[v]
        for v in block:
            next_residual[v] = 0.0
        details.append({"index": index + 1, "nodes": list(block),
                        "parallel_nodes": sorted(parallel),
                        "residual_before": dict(residual),
                        "node_analysis": node_details, "paths": paths,
                        "path_statistics": path_stats,
                        "response_bound": block_bound,
                        "node_completion_bound": node_completion,
                        "residual_after": dict(next_residual)})
        residual = next_residual

    tail_length = math.fsum(dag.wcet[v] for v in tail)
    bound = math.fsum(b["response_bound"] for b in details) + tail_length  # (18)
    if not math.isfinite(bound):
        raise ValueError("Analysis overflow; rescale the WCETs")
    lower_bound = max(dag.critical_path()[1], dag.workload() / cores)
    issues = []
    if bound + 1e-10 * max(1, lower_bound) < lower_bound:
        issues.append("Candidate is below the workload/critical-path lower bound")
    if observed_makespan is not None:
        if not math.isfinite(observed_makespan) or observed_makespan <= 0:
            raise ValueError("observed_makespan must be finite and positive")
        if bound + 1e-10 * max(1, observed_makespan) < observed_makespan:
            issues.append("Candidate is below the observed makespan: counterexample detected")
    certified, certified_method = certified_response_bound(dag, cores)
    return {"bound": bound, "status": ("counterexample_detected" if issues
                                       else "experimental_unverified"),
            "is_proven_upper_bound": False,
            "model_safe_bound": graham_bound(dag, cores),
            "certified_response_bound": certified, "certified_response_method": certified_method,
            "necessary_lower_bound": lower_bound, "sanity_issues": issues,
            "observed_makespan": observed_makespan,
            "concurrent_nodes": {v: sorted(concurrent[v]) for v in dag.order},
            "blocks": details, "tail": list(tail), "tail_length": tail_length,
            "assumptions": ["One single-source/single-sink DAG release",
                            "Identical cores; no communication costs",
                            "Non-preemptive nodes; canonical Algorithms 1/2 priorities",
                            "Reproduced recurrence has known counterexamples; use model_safe_bound for guarantees",
                            "Reduced residual estimates are not treated as proof of an already-running node"]}


def calculate_algorithm_3_bound(dag, blocks, tail, local_paths_list, priorities, cores):
    """Backward-compatible EXPERIMENTAL candidate API; not a safety guarantee.

    Use analyze_algorithm_3 for diagnostics and its model_safe_bound field
    when a guaranteed bound under the baseline model is needed.
    """
    return analyze_algorithm_3(dag, blocks, tail, local_paths_list, priorities, cores)["bound"]


def analyze_timing(dag, blocks, tail, priorities, cores, *, observed_makespan=None,
                   max_states=200000):
    """Exact compact path evaluation of the experimental recurrence.

    No 10,000-path cap. If interference STATE memory exceeds max_states, keep
    the independent certified bound and report that the candidate is unevaluated.
    """
    check_cores(cores)
    try:
        result = analyze_algorithm_3(dag, blocks, tail, None, priorities, cores,
                                    observed_makespan=observed_makespan, max_states=max_states)
    except AnalysisStateLimit as error:
        certified, method = certified_response_bound(dag, cores)
        return {'bound': None, 'status': 'not_evaluated_state_budget', 'is_proven_upper_bound': False,
                'model_safe_bound': graham_bound(dag, cores),
                'certified_response_bound': certified, 'certified_response_method': method,
                'necessary_lower_bound': max(dag.critical_path()[1], dag.workload() / cores),
                'sanity_issues': [str(error)], 'observed_makespan': observed_makespan,
                'local_paths': None, 'blocks': [], 'tail': list(tail),
                'tail_length': sum(dag.wcet[v] for v in tail), 'path_evaluation': 'compact_interference_dp'}
    result['local_paths'] = None  # Full lists are intentionally not materialized.
    result['path_evaluation'] = 'compact_interference_dp'
    return result
