import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Import Line2D for legend
import random

# Define the graph structure
graph = {
    "Pancham Society": ["Andheri Recreation Club"],
    "Andheri Recreation Club": ["Navrang Cinema"],
    "Navrang Cinema": ["Sagar Shopping Centre"],
    "Sagar Shopping Centre": ["Cambridge Institute"],
    "Cambridge Institute": ["Al Zaika Restaurant"],
    "Al Zaika Restaurant": ["Benchmarx Academy", "M.p.Traders"],
    "Benchmarx Academy": ["PEST MANAGEMENT & SERVICES"],
    "PEST MANAGEMENT & SERVICES": ["MVLU"],
    "M.p.Traders": ["Mahakali Book Center"],
    "Mahakali Book Center": ["MVLU"],
    "MVLU": []  # End point
}

# Randomly assign weights to the edges
weighted_graph = {}
for node, neighbors in graph.items():
    weighted_graph[node] = [(neighbor, random.randint(1, 10)) for neighbor in neighbors]

# Print the weighted graph
print("Weighted Graph:")
for node, neighbors in weighted_graph.items():
    print(f"{node}: {neighbors}")

# Heuristic function (arbitrary values for demonstration)
def h(node):
    heuristic_values = {
        "Pancham Society": 10,
        "Andheri Recreation Club": 8,
        "Navrang Cinema": 6,
        "Sagar Shopping Centre": 4,
        "Cambridge Institute": 2,
        "Al Zaika Restaurant": 5,
        "Benchmarx Academy": 3,
        "PEST MANAGEMENT & SERVICES": 2,
        "M.p.Traders": 1,
        "Mahakali Book Center": 1,
        "MVLU": 0
    }
    return heuristic_values.get(node, float('inf'))

# A* algorithm implementation
def aStarAlgo(start_node, stop_node):
    open_set = {start_node}
    closed_set = set()
    g = {start_node: 0}
    parents = {start_node: start_node}

    while open_set:
        n = None

        for v in open_set:
            if n is None or g[v] + h(v) < g[n] + h(n):
                n = v

        if n == stop_node:
            path = []
            total_weight = 0  # To calculate total weight of the path
            while parents[n] != n:
                path.append(n)
                total_weight += next(weight for neighbor, weight in weighted_graph[parents[n]] if neighbor == n)
                n = parents[n]
            path.append(start_node)
            path.reverse()
            print('Path found: {}'.format(path))
            print('Total weight of the path: {}'.format(total_weight))
            return path

        open_set.remove(n)
        closed_set.add(n)

        for m, weight in weighted_graph.get(n, []):
            if m not in open_set and m not in closed_set:
                open_set.add(m)
                parents[m] = n
                g[m] = g[n] + weight  # Use the random weight
            elif g[m] > g[n] + weight:
                g[m] = g[n] + weight
                parents[m] = n
                if m in closed_set:
                    closed_set.remove(m)
                    open_set.add(m)

    print('Path does not exist!')
    return None

# Run A* Search
path = aStarAlgo("Pancham Society", "MVLU")

# Create a directed graph for visualization
G = nx.DiGraph()

# Add nodes to the graph
for node in weighted_graph:
    G.add_node(node)

# Add edges to the graph with weights
for node, neighbors in weighted_graph.items():
    for neighbor, weight in neighbors:
        G.add_edge(node, neighbor, weight=weight)

# Use spring layout for better positioning
pos = nx.spring_layout(G, seed=10)  # Use a seed for reproducibility

# Draw the graph with labels
plt.figure(figsize=(10, 8))
# Draw all nodes and edges with labels
nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', font_size=10)

# Highlight the path in red
if path:
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='red')
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

# Draw edge weights
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Add heuristic values as labels in the format 'Node_Name: Heuristic_Value'
heuristic_labels = {node: f"{node}: {h(node)}" for node in graph}
nx.draw_networkx_labels(G, pos, labels=heuristic_labels, font_color='black', verticalalignment='bottom')

# Add legend for the A* path
legend_elements = [
    Line2D([0], [0], color='red', lw=2, label='A* Path Edges'),
    Line2D([0], [0], marker='o', color='red', label='A* Path Nodes', markersize=10)
]
plt.legend(handles=legend_elements, loc='upper left')

plt.title('Graph Representation with A* Path Highlighted')
plt.axis('off')
plt.show()
