def he2019_priorities(dag):
    dist_from_src = {v: dag.wcet[v] for v in dag.order}
    for v in dag.order:
        if dag.pred[v]:
            dist_from_src[v] = max(dist_from_src[p] for p in dag.pred[v]) + dag.wcet[v]
            
    dist_to_sink = {v: dag.wcet[v] for v in dag.order}
    for v in reversed(dag.order):
        if dag.succ[v]:
            dist_to_sink[v] = max(dist_to_sink[s] for s in dag.succ[v]) + dag.wcet[v]
            
    max_path_thru = {}
    for v in dag.order:
        max_path_thru[v] = dist_from_src[v] + dist_to_sink[v] - dag.wcet[v]
        
    sorted_nodes = sorted(dag.order, key=lambda v: (max_path_thru[v], -dag.order.index(v)), reverse=True)
    
    priorities = {}
    p = len(dag.wcet)
    for v in sorted_nodes:
        priorities[v] = p
        p -= 1
        
    return priorities


def zhao2020_priorities(dag):
    cp_nodes_list, _ = dag.critical_path()
    cp_set = set(cp_nodes_list)
    
    priorities = {}
    p = len(dag.wcet)

    for v in cp_nodes_list:
        priorities[v] = p
        p -= 1
        
    for v in dag.order:
        if v not in cp_set:
            priorities[v] = p
            p -= 1
            
    return priorities