import networkx as nx
import matplotlib.pyplot as plt

# Define the graph
graph = {
    "h.m.p school staff quarters": ["vrundavan hotel"],
    "vrundavan hotel": ["new dadabhai road"],
    "new dadabhai road": ["andheri market"],
    "andheri market": ["jp road"],
    "jp road": ["andheri subway"],
    "andheri subway": ["mogra village", "old nagardas rd"],
    "mogra village": ["post office"],
    "post office": ["MVLU"],
    "old nagardas rd": ["andheri metro station"],
    "andheri metro station": ["MVLU"],
    "MVLU": []  # End point
}

# DFS function to find the path from source to destination
def depth_first_search(graph, source, destination):
    stack = [(source, [source])]
    visited = set()

    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        if node == destination:
            return path  # Return the path if destination is reached

        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                stack.append((neighbour, path + [neighbour]))

    return None  # Return None if no path is found

# Run DFS to find the path from "h.m.p school staff quarters" to "MVLU"
dfs_path = depth_first_search(graph, "h.m.p school staff quarters", "MVLU")
print("DFS Path:", dfs_path)

# Create and draw the graph
G = nx.DiGraph(graph)  # Use nx.DiGraph for a directed graph
pos = nx.spring_layout(G, seed=6)  # Position the nodes using a spring layout

# Draw the graph
nx.draw(G, pos, node_size=300, font_size=10, font_color='black',
        with_labels=True, node_color='skyblue', edge_color='gray')

# Highlight the DFS path on the graph
if dfs_path:
    path_edges = list(zip(dfs_path, dfs_path[1:]))  # Create a list of edges in the path
    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                           edge_color='blue', width=2)  # Highlight the path edges in blue
    nx.draw_networkx_nodes(G, pos, nodelist=dfs_path,
                           node_color='blue', node_size=300)  # Highlight the path nodes in blue

# Add a legend to the plot
plt.legend(handles=[
    plt.Line2D([0], [0], color='gray', label='Normal Path'),
    plt.Line2D([0], [0], color='blue', label='DFS Path')
], loc='upper right')

# Show the plot
plt.show()
