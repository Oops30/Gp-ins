import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

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

# BFS function to find the shortest path from source to goal
def breadth_first_search(graph, source, goal):
    queue = deque([[source]])
    visited = {source}

    while queue:
        path = queue.popleft()
        current = path[-1]

        # If the goal is reached, return the path
        if current == goal:
            return path

        # Explore the neighbors
        for neighbour in graph.get(current, []):
            if neighbour not in visited:
                visited.add(neighbour)
                new_path = path + [neighbour]  # Create a new path
                queue.append(new_path)

    return None  # Return None if no path is found

# Run BFS to get the shortest path from "h.m.p school staff quarters" to "MVLU"
bfs_path = breadth_first_search(graph, "h.m.p school staff quarters", "MVLU")
print("BFS Path:", bfs_path)

# Create and draw the graph
G = nx.DiGraph(graph)  # Use nx.DiGraph for a directed graph
pos = nx.spring_layout(G, seed=6)  # Position the nodes using a spring layout

# Draw the graph
nx.draw(G, pos, node_size=300, font_size=10, font_color='black',
        with_labels=True, node_color='skyblue', edge_color='grey')

# Highlight the BFS path on the graph
if bfs_path:
    path_edges = list(zip(bfs_path, bfs_path[1:]))  # Create a list of edges in the path
    nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                           edge_color='red', width=2)  # Highlight the path edges in red
    nx.draw_networkx_nodes(G, pos, nodelist=bfs_path,
                           node_color='red', node_size=300)  # Highlight the path nodes in red

# Add a legend to the plot
plt.legend(handles=[
    plt.Line2D([0], [0], color='grey', label='Normal Path'),
    plt.Line2D([0], [0], color='red', label='BFS Path')
], loc='upper right')

# Show the plot
plt.show()
