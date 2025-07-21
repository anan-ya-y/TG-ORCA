import rustworkx as rx
import rustworkx.visualization
import numpy as np
import json, os

class Graph:
    def __init__(self, directed=False, weightfn=None):
        if not directed:
            self.graph = rx.PyGraph()
        else:
            self.graph = rx.PyDiGraph()
        self.directed = directed
        self.vertices = {} # 3d point to vertex id
        
        if weightfn is None:
            # euclidean distance. 
            self.weightfn = lambda x, y: np.linalg.norm(np.array(x) - np.array(y))
        else:
            self.weightfn = weightfn

    def reset(self):
        self.graph = rx.PyGraph()
        self.vertices = {}

    ## Add vertex WITHOUT collision checking
    def add_vertex(self, v):
        if type(v) is list:
            v = tuple(v)
        if v in self.vertices:
            return self.vertices[v], False
        vid = self.graph.add_node(v)
        self.vertices[v] = vid
        return vid, True
    
    def add_many_vertices(self, vertices):
        for v in vertices:
            self.add_vertex(v)

    # Also removes all neighboring edges. 
    def remove_vertex(self, v):
        vid = self.vertices[v]
        neighbors = self.graph.neighbors(vid)
        for n in neighbors:
            self.graph.remove_edge(vid, n)
        self.graph.remove_node(vid)
        del self.vertices[v]
    
    ## Add edge, no collision checking
    ## both endpoints must already be inside the graph.
    def add_edge(self, c1, c2, edge_weight=None):
        if edge_weight is None:
            edge_weight = self.weightfn(c1, c2)
        self.graph.add_edge(self.vertices[c1], self.vertices[c2], edge_weight)

    def remove_edge(self, c1, c2):
        self.graph.remove_edge(self.vertices[c1], self.vertices[c2])

    def get_shortest_path(self, start, goal):
        source = self.vertices[start]
        target = self.vertices[goal]
        if self.directed:
            path = rustworkx.digraph_dijkstra_shortest_paths(graph=self.graph, 
                                                    source=source, 
                                                    target=target, 
                                                    weight_fn=lambda x:x)
        else:
            path = rustworkx.graph_dijkstra_shortest_paths(graph=self.graph, 
                                                    source=source, 
                                                    target=target, 
                                                    weight_fn=lambda x:x)
        if target in path:
            path = path[target]
        else:
            path = []
        return [self.graph.get_node_data(node) for node in path]

    # Find the nearest num_neighbors points to this one, that is at MOST radius dist away. 
    def find_nearest(self, point, num_neighbors, radius=-1):
        # Get list of nodes
        nodes = self.graph.nodes()

        # We do not want cfg to show up in our list of neighbors. 
        try:
            nodes.remove(point)
        except:
            pass

        # Create list of cfgs and distances
        neighbors = [(v, self.weightfn(point, v)) for v in nodes]

        if radius != -1:
            # Filter by distance
            neighbors = list(filter(lambda w : w[1] <= radius, neighbors))

        # Sort list by distances
        neighbors.sort(key=lambda w: w[1])

        # Return the num_neighbors nearest neighbors
        return [v[0] for v in neighbors[0:num_neighbors]]
    
    # Find the nearest num_neighbors points to this one, that is at LEAST radius dist away. 
    def k_nearest_neighbors(self, point, num_neighbors, min_dist=0):
        # Get list of nodes
        nodes = self.graph.nodes()

        # We do not want cfg to show up in our list of neighbors. 
        try:
            nodes.remove(point)
        except:
            pass

        # Create list of cfgs and distances
        neighbors = [(v, self.weightfn(point, v)) for v in nodes]

        # Filter by distance
        neighbors = list(filter(lambda w : w[1] >= min_dist, neighbors))

        # Sort list by distances
        neighbors.sort(key=lambda w: w[1])

        # Return the num_neighbors nearest neighbors
        return [v[0] for v in neighbors[0:num_neighbors]]

    def dist_to_nearest_vertex(self, point):
        # return min([self.weightfn(point, v) for v in self.graph.nodes()])
        nearest = self.find_nearest(point, 1)
        if len(nearest) == 0:
            return np.inf
        return self.weightfn(point, nearest[0])
    
    def num_connected_components(self):
        return rx.connected_components(self.graph)
    

    def get_nodes(self):
        return self.graph.nodes() # gets the list of DATA of nodes
    def get_vertices(self):
        return self.get_nodes()
    
    def get_edges(self):
        return [(self.graph.get_node_data(s), self.graph.get_node_data(t)) \
                            for s, t in self.graph.edge_list()]

    # returns [(source, target, weight), ...]
    def get_edges_and_weights(self):
        return [(self.graph.get_node_data(s), self.graph.get_node_data(t), w) \
                            for s, t, w in self.graph.weighted_edge_list()]
    
    def save_to_json(self, filename):
        data = {
            "vertices": [list(v) for v in self.vertices],
            "edges": [[list(s), list(t), w] for s, t, w in self.get_edges_and_weights()]
        }

        with open(filename, "w") as f:
            json.dump(data, f)

    def load_from_json(self, filename):
        if not os.path.exists(filename):
            raise ValueError(f"Topological Graph error: {filename} file not found.")
        
        with open(filename, "r") as f:
            data = json.load(f)
        self.graph = rx.PyGraph()
        self.add_many_vertices(data["vertices"])
        for s, t, w in data["edges"]:
            s = tuple(s)
            t = tuple(t)
            self.add_edge(s, t, edge_weight=w)

    # ============================== PLOTTING ============================== #
    def plot_graph(self, ax, color="blue"):
        pos = {self.vertices[node]: (node[0], node[1]) for node in self.graph.nodes()}
        rx.visualization.mpl_draw(self.graph, 
                                pos=pos, 
                                ax=ax, 
                                node_size=5, 
                                node_color='black', 
                                width=1, 
                                edge_color=color)

    def plot_subgraph(self, graph, ax, color="green"):
        pos = {self.vertices[node]: (node[0], node[1]) for node in graph.nodes()}
        rx.visualization.mpl_draw(graph, 
                                pos=pos, 
                                ax=ax, 
                                node_size=5, 
                                node_color=color, 
                                width=1, 
                                edge_color=color)