from itertools import combinations

def max_antichain_size(node_set, k, dag, C):
    valid = [v for v in node_set if C[k].get(v, 0) > 0]
    if not valid: return 0
    
    for size in range(len(valid), 0, -1):
        for sub in combinations(valid, size):
            indep = True
            for i in range(len(sub)):
                for j in range(i+1, len(sub)):
                    u, v = sub[i], sub[j]
                    if v in dag.ancestors.get(u, set()) or u in dag.ancestors.get(v, set()):
                        indep = False
                        break
                if not indep: break
            if indep:
                return size
    return 0

def calculate_algorithm_3_bound(dag, blocks, tail, local_paths_list, priorities, cores):
    descendants = {v: set() for v in dag.wcet}
    for v in reversed(dag.order):
        descendants[v] = set(dag.succ[v])
        for u in dag.succ[v]:
            descendants[v].update(descendants[u])
            
    con = {}
    all_nodes = set(dag.wcet.keys())
    for v in dag.wcet:
        con[v] = all_nodes - dag.ancestors.get(v, set()) - descendants[v] - {v}
        
    C = {1: {v: dag.wcet[v] for v in dag.wcet}}
    R_node = {}
    I_e = {}
    I_hp_all = {}
    I_lp_all = {}
    
    R_block_times = []
    
    for k_idx, block in enumerate(blocks):
        k = k_idx + 1
        B_k = set(block)
        
        later_blocks = set()
        for l in range(k_idx + 1, len(blocks)):
            later_blocks.update(blocks[l])
        later_blocks.update(tail)
        
        para_B_k = set()
        for v in B_k:
            para_B_k.update(con[v])
        para_B_k = para_B_k.intersection(later_blocks)
        
        I_e[k] = {}
        R_node[k] = {}
        I_hp_all[k] = {}
        I_lp_all[k] = {}
        
        nodes_in_scope = [v for v in dag.order if v in B_k or v in para_B_k]
        
        for i in nodes_in_scope:
            if C[k].get(i, 0) == 0:
                R_node[k][i] = 0
                I_e[k][i] = set()
                I_hp_all[k][i] = set()
                I_lp_all[k][i] = set()
                continue
                
            anc_I_e = set()
            for anc in dag.ancestors.get(i, set()):
                if anc in I_e[k]:
                    anc_I_e.update(I_e[k][anc])
                    
            I_hp, I_lp = set(), set()
            for j in (B_k | para_B_k):
                if j in con[i] and j not in anc_I_e and C[k].get(j, 0) > 0:
                    if priorities[j] > priorities[i]:
                        I_hp.add(j)
                    elif priorities[j] < priorities[i]:
                        I_lp.add(j)
                        
            I_hp_all[k][i] = I_hp
            I_lp_all[k][i] = I_lp
            
            lp_sorted = sorted(list(I_lp), key=lambda j: C[k][j], reverse=True)
            I_e[k][i] = I_hp.union(set(lp_sorted[:cores-1]))
            
            max_pred_R = max((R_node[k][p] for p in dag.pred[i] if p in R_node[k]), default=0)
                    
            if max_antichain_size(I_e[k][i], k, dag, C) <= cores - 1:
                delay = 0
            else:
                delay = sum(C[k][j] for j in I_e[k][i]) / cores
                
            R_node[k][i] = max_pred_R + C[k][i] + delay
            
        R_paths = []
        for path in local_paths_list[k_idx]:
            path_I_hp, path_I_lp = set(), set()
            path_W = sum(C[k][v] for v in path if C[k].get(v, 0) > 0)
            
            for v in path:
                if v in I_hp_all[k]: path_I_hp.update(I_hp_all[k][v])
                if v in I_lp_all[k]: path_I_lp.update(I_lp_all[k][v])
                    
            lp_sorted_path = sorted(list(path_I_lp), key=lambda j: C[k][j], reverse=True)
            path_I_e = path_I_hp.union(set(lp_sorted_path[:cores-1]))
            
            if max_antichain_size(path_I_e, k, dag, C) <= cores - 1:
                delay = 0
            else:
                delay = sum(C[k][j] for j in path_I_e) / cores
                
            R_paths.append(path_W + delay)
            
        R_k = max(R_paths) if R_paths else 0
        R_block_times.append(R_k)

        if k < len(blocks):
            C[k+1] = dict(C[k])
            max_R_Bk = max((R_node[k][l] for l in B_k if l in R_node[k]), default=0)
            for i in para_B_k:
                C[k+1][i] = min(dag.wcet[i], max(0, R_node[k][i] - max_R_Bk))
            for i in B_k:
                C[k+1][i] = 0 
                
    tail_len = sum(dag.wcet[v] for v in tail)
    return sum(R_block_times) + tail_len
