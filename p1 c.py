import networkx as nx
import matplotlib.pyplot as plt

# Define the same graph
graph = {
    "Mukund Hospital": ["Andheri Kurla road"],
    "Andheri Kurla road": ["Andheri Ghatkopar road", "corporate park"],
    "Andheri Ghatkopar road": ["Bajaj"],
    "Bajaj": ["Highway"],
    "Highway": ["MVLU"],
    "corporate park": ["silver"],
    "silver": ["bisleri"],
    "bisleri": ["Highway"],
    "MVLU": []
}

# BFS function to find the path from source to destination
def breadth_first_search(graph, source, destination):
    queue = [(source, [source])]
    visited = set()
    while queue:
        (node, path) = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if node == destination:
            return path  # Return the path if destination is reached
        for neighbor in graph[node]:
            queue.append((neighbor, path + [neighbor]))
    return None  # Return None if no path is found

# DFS function to find the path from source to destination
def depth_first_search(graph, source, destination):
    stack = [(source, [source])]
    visited = set()
    while stack:
        (node, path) = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node == destination:
            return path  # Return the path if destination is reached
        for neighbor in graph[node]:
            stack.append((neighbor, path + [neighbor]))
    return None  # Return None if no path is found

# Run BFS and DFS to find the path from "Mukund Hospital" to "MVLU"
bfs_path = breadth_first_search(graph, "Mukund Hospital", "MVLU")
dfs_path = depth_first_search(graph, "Mukund Hospital", "MVLU")

print("BFS Path:", bfs_path)
print("DFS Path:", dfs_path)

# Create a directed graph using networkx
G = nx.DiGraph(graph)

# Draw the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G,seed=6)  # Position the nodes using a spring layout
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray')

# Highlight the BFS path
if bfs_path:
    edge_list_bfs = [(bfs_path[i], bfs_path[i + 1]) for i in range(len(bfs_path) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=edge_list_bfs, edge_color='blue', width=2, label="BFS Path")

# Highlight the DFS path
if dfs_path:
    edge_list_dfs = [(dfs_path[i], dfs_path[i + 1]) for i in range(len(dfs_path) - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=edge_list_dfs, edge_color='green', width=2, style="dashed", label="DFS Path")

# Add legend for BFS and DFS paths
plt.legend(["BFS Path", "DFS Path"])
plt.title("Graph Visualization with BFS and DFS Paths")
plt.show()
