import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

class DAGGenerator:
    def __init__(self, min_depth=5, max_depth=8, max_parallelism=4):
        self.depth = random.randint(min_depth, max_depth)
        self.max_parallelism = max_parallelism
        self.G = nx.DiGraph()

    def generate_dag(self):
        node_id = 1
        layers = []
        
        for layer in range(self.depth):
            num_nodes = random.randint(1, self.max_parallelism)
            current_layer_nodes = []
            
            for _ in range(num_nodes):
                wcet = random.randint(150, 250) 
                num_temp_samples = wcet // 10
                temp_trace = np.round(np.random.uniform(50.0, 60.0, num_temp_samples), 2).tolist()
                self.G.add_node(node_id, 
                                layer=layer, 
                                wcet=wcet, 
                                temp_trace=temp_trace)
                current_layer_nodes.append(node_id)
                node_id += 1
                
            layers.append(current_layer_nodes)
            
        for i in range(len(layers) - 1):
            for u in layers[i]:
                v = random.choice(layers[i+1])
                self.G.add_edge(u, v)
                
                for other_v in layers[i+1]:
                    if other_v != v and random.random() > 0.5:
                        self.G.add_edge(u, other_v)
                        
        return self.G

    def visualize(self):
        plt.figure(figsize=(10, 6))
        pos = nx.multipartite_layout(self.G, subset_key="layer")
        nx.draw_networkx_nodes(self.G, pos, node_color='lightblue', node_size=600)
        nx.draw_networkx_edges(self.G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
        nx.draw_networkx_labels(self.G, pos, font_size=10, font_weight='bold')
        
        plt.title(f"Synthetic DAG Task (Depth: {self.depth})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    generator = DAGGenerator(min_depth=5, max_depth=8, max_parallelism=4)
    dag = generator.generate_dag()
    
    print("Nodes and WCET Dataset")
    for node, data in list(dag.nodes(data=True))[:3]:
        print(f"Node {node}: WCET = {data['wcet']} ms")
        print(f"Temp Trace (first 5 samples): {data['temp_trace'][:5]} °C\n")
    
    generator.visualize()
