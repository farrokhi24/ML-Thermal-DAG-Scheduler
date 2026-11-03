import random
import networkx as nx
import json
import matplotlib.pyplot as plt

def uunifast_discard(n, total_workload):
    while True:
        sum_w = total_workload
        vect_w = []
        for i in range(1, n):
            next_sum_w = sum_w * (random.random() ** (1.0 / (n - i)))
            vect_w.append(sum_w - next_sum_w)
            sum_w = next_sum_w
        vect_w.append(sum_w)
        
        vect_w_int = [int(round(w)) for w in vect_w]
        diff = int(total_workload - sum(vect_w_int))
        
        for _ in range(abs(diff)):
            idx = random.randint(0, n - 1)
            vect_w_int[idx] += 1 if diff > 0 else -1
            
        if all(w > 0 for w in vect_w_int):
            return vect_w_int

class DAGGenerator:
    def __init__(self, seed=2026, total_workload=500, min_depth=5, max_depth=8, max_parallelism=4):
        self.seed = seed
        random.seed(self.seed)
        self.total_workload = total_workload
        self.depth = random.randint(min_depth, max_depth)
        self.max_parallelism = max_parallelism
        self.G = nx.DiGraph()

    def generate_dag(self):
        node_id = 1
        layers = []
        
        layers.append([node_id])
        self.G.add_node(node_id, layer=0)
        node_id += 1
        
        for layer in range(1, self.depth - 1):
            num_nodes = random.randint(2, self.max_parallelism)
            current_layer = []
            for _ in range(num_nodes):
                self.G.add_node(node_id, layer=layer)
                current_layer.append(node_id)
                node_id += 1
            layers.append(current_layer)
            
        layers.append([node_id])
        self.G.add_node(node_id, layer=self.depth - 1)
        node_id += 1
        
        for i in range(len(layers) - 1):
            for u in layers[i]:
                v = random.choice(layers[i+1])
                self.G.add_edge(u, v)
                
                for other_v in layers[i+1]:
                    if other_v != v and random.random() < 0.2:
                        self.G.add_edge(u, other_v)

        num_nodes = len(self.G.nodes)
        wcet_list = uunifast_discard(num_nodes, self.total_workload)
        
        wcet_dict = {}
        for i, node in enumerate(self.G.nodes):
            wcet_dict[node] = wcet_list[i]
            self.G.nodes[node]['wcet'] = wcet_list[i]
            
        return wcet_dict, list(self.G.edges)

    def export_json(self, filepath):
        wcet_dict, edges = self.generate_dag()
        data = {
            "name": f"Generated DAG (Seed: {self.seed}, Workload: {self.total_workload})",
            "time_unit": "ms",
            "source": "Generated via UUniFast-discard",
            "wcet": wcet_dict,
            "edges": [list(e) for e in edges]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def visualize(self):
        plt.figure(figsize=(10, 6))
        pos = nx.multipartite_layout(self.G, subset_key="layer")
        nx.draw_networkx_nodes(self.G, pos, node_color='lightblue', node_size=600)
        nx.draw_networkx_edges(self.G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
        nx.draw_networkx_labels(self.G, pos, font_size=10, font_weight='bold')
        plt.title(f"Synthetic DAG Task (Depth: {self.depth}, Workload: {self.total_workload})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    generator = DAGGenerator(seed=2026, total_workload=500, max_parallelism=4)
    generator.export_json("examples/generated_dag_uunifast.json")
    generator.visualize()
