import numpy as np
import matplotlib.pyplot as plt
import json, os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import nearest_points
from matplotlib.patches import Polygon as Poly
try:
    from crowd_sim.envs.utils.graph import Graph
    # from crowd_sim.envs.skeleton_to_graph import skeleton_to_graph
except:
    from graph import Graph


class TopologicalGraph(Graph):
    def __init__(self, obstacles_vertices, environment_vertices, \
                        type="prm_maprm_combined", saveFile=True, saveImage=True):    
        super().__init__(directed=False)

        self.obstacles_vertices = obstacles_vertices
        self.environment_vertices = environment_vertices

        self.env_polygon = Polygon(environment_vertices + [environment_vertices[0]])
        self.obstacle_polygons = [Polygon(obstacle+[obstacle[0]]) for obstacle in obstacles_vertices]
        self.env_bounds = {
            "xmin": min([x for x, y in environment_vertices]),
            "xmax": max([x for x, y in environment_vertices]),
            "ymin": min([y for x, y in environment_vertices]),
            "ymax": max([y for x, y in environment_vertices])
        }

        self.validity_cache = {} # point -> buffer distance. if -1, then invalid.
        # np.random.seed(0)

        print("Calculating top graph...")
        self.calculate_topological_graph(type, saveFile=saveFile, saveImage=saveImage)


    # returns [] if point is invalid. 
    def __get_nearest_obstacles(self, point, n=2):
        if not self.env_polygon.contains(Point(point)):
            return []
        obstacle_dists = []
        for obstacle in self.obstacle_polygons:
            if obstacle.contains(Point(point)):
                return []
            else:
                obstacle_dists.append(obstacle.distance(Point(point)))
        obstacle_dists = [self.env_polygon.exterior.distance(Point(point))] + obstacle_dists

        sorted_indices = np.argsort(obstacle_dists)[:n]
        all_objects = [self.env_polygon] + self.obstacle_polygons
        return [all_objects[i] for i in sorted_indices]  

    def __is_valid(self, point, buffer_dist, debug=False):
        if debug:
            import pdb; pdb.set_trace()
        cache_key = tuple(point)
        if tuple(point) in self.validity_cache:
            if debug:
                print(f"Distance for {point} is {self.validity_cache[cache_key]}, buffer={buffer_dist}")
            return self.validity_cache[cache_key] >= buffer_dist
        
        if not self.env_polygon.contains(Point(point)):
            self.validity_cache[cache_key] = -1 # invalid
            if debug:
                print(f"Point {point} not valid. ")
            return False
        
        dist = self.env_polygon.exterior.distance(Point(point))

        for obstacle in self.obstacle_polygons:
            if obstacle.contains(Point(point)):
                self.validity_cache[cache_key] = -1
                if debug:
                    print(f"Point {point} not valid. ")
                return False
            dist = min(dist, obstacle.distance(Point(point)))
    
        self.validity_cache[cache_key] = dist   
        if debug:
            print(f"Distance for {point} is {self.validity_cache[cache_key]}, buffer={buffer_dist}")
        return dist >= buffer_dist

    # returns tuple, (True, [end]) if whole edge is valid. 
    # returns tuple, (False, [x, y]) if whole edge is not valid, where [x, y] is farthest point. 
    def __edge_valid(self, start:tuple, end:tuple, buffer_dist=0, resolution=0.05):
        if not self.__is_valid(start, buffer_dist) or \
            not self.__is_valid(end, buffer_dist):
            return False, []

        length = np.linalg.norm(np.array(start) - np.array(end))
        numpoints = max(2, int(length / resolution)) 
        x_points = np.linspace(float(start[0]), float(end[0]), numpoints)
        y_points = np.linspace(float(start[1]), float(end[1]), numpoints)
        farthest_point = [(x_points[0], y_points[0])]
        for (x, y) in zip(x_points, y_points):
            p = (x, y)
            if not self.__is_valid(p, buffer_dist):
                return False, farthest_point
            farthest_point = p

        return True, farthest_point

    def __sample_point(self, sample_res):
        return np.random.choice(np.arange(self.env_bounds["xmin"], self.env_bounds["xmax"], sample_res)), \
               np.random.choice(np.arange(self.env_bounds["ymin"], self.env_bounds["ymax"], sample_res))

    def __push_sample_to_ma(self, sample):
        nearest_obstacles = self.__get_nearest_obstacles(sample, n=2)
        if nearest_obstacles == []: # sample was not valid. 
            return []

        near_points = [nearest_points(Point(sample), obs.exterior)[1] \
                        for obs in nearest_obstacles]
        near_points = [(p.x, p.y) for p in near_points]
        c = np.average(near_points, axis=0)

        # find the line between sample and c (=l)
        # new point = farthest point along line l
        _, v = self.__edge_valid(sample, c, buffer_dist=0)
        return v

    def __connect_vertex_to_graph(self, sample, k_neighbors, knn_dist=0, buffer_dist=0):
        edges = []
        neighbors = self.k_nearest_neighbors(sample, k_neighbors, min_dist=knn_dist)
        
        for neighbor in neighbors:
            edge_valid, _ = self.__edge_valid(sample, neighbor, buffer_dist=buffer_dist)
            if edge_valid:
                edges.append((sample, neighbor))

        if edges == []: # don't add this sample. 
            return False
        
        
        # add this sample to graph. 
        for e in edges:
            self.add_edge(e[0], e[1])
        return True

    def get_random_valid_point(self, buffer_dist=0.15, ntries=1000):
        for _ in range(ntries):
            point = (np.random.uniform(self.env_bounds["xmin"], self.env_bounds["xmax"]), \
                     np.random.uniform(self.env_bounds["ymin"], self.env_bounds["ymax"]))
            if self.__is_valid(point, buffer_dist):
                return point
        return None

    def query_graph_with_augmentation(self, start, goal):
        # print("Querying graph...")
        # Add start and goal to the graph.
        all_verts = self.get_vertices()
        _, add_start = self.add_vertex(start)
        _, add_goal = self.add_vertex(goal)

        # Add dummy edges 
        for vert in all_verts:
            if self.__edge_valid(start, vert, buffer_dist=0.1)[0]:
                self.add_edge(start, vert)
            if self.__edge_valid(goal, vert, buffer_dist=0.1)[0]:
                self.add_edge(goal, vert)
        
        # query. 
        path = self.get_shortest_path(start, goal)
        path = self.path_shortcut(path, buffer=0.1)

        # remove the augmentations. 
        if add_start:
            self.remove_vertex(start)
        if add_goal:
            self.remove_vertex(goal)

        # assert path != []
        # print("Done querying graph.")
        return path


    '''
    This is a modified version of PRM that samples points with high "buffer" (clearance) first and 
    then decreases the buffer as time passes. 

    # Parameters (and their defaults)
    v = 30 # min num of nodes
    start_buffer = 4 # starting buffer dist from obstacles
    buffer_step = 0.5 # how much to decrease buffer  each time.
    n = 20 # number of samples per iteration
    n_attempts = 200 # number of times to try sampling
    k = 8 # number of nearest neighbors to connect to
    knn_dist = 0.5 # min distance to connect to nearest neighbors
    sample_res = 0.25 # resolution of sampling
    sample_sparsity = 1.5 # min distance between samples.
    '''
    def prm(self, v=40, n=20, n_attempts=200, k=8, knn_dist=0.5, \
            sample_res=0.25, sample_sparsity=1.5, start_buffer=4, buffer_step=0.5):
        buffer = start_buffer

        while True:
            if buffer < 0:
                break
            if self.num_connected_components() == 1 and len(self.vertices.keys()) >= v:
                break

            # Sample n points inside env.
            samples = set()
            for _ in range(n_attempts):
                if len(samples) == n:
                    break
                sample = self.__sample_point(sample_res)
                if self.__is_valid(sample, buffer):
                    if self.dist_to_nearest_vertex(sample) > sample_sparsity:
                        samples.add(sample)
                        self.add_vertex(sample)


            # add the edges. 
            for node in samples:
                has_neighbors = self.__connect_vertex_to_graph(node, k, knn_dist, max(buffer, 0.25))
                if not has_neighbors:
                    self.remove_vertex(node)

            buffer -= buffer_step

    def grid(self, resolution=0.75):
        X, Y = np.mgrid[self.env_bounds["xmin"]:self.env_bounds["xmax"]:resolution, \
                        self.env_bounds["ymin"]:self.env_bounds["ymax"]:resolution]
        positions = np.vstack([X.ravel(), Y.ravel()])
        positions = list((x, y) for x, y in zip(positions[0], positions[1]))

        positions = [p for p in positions if self.__is_valid(p, buffer_dist=resolution)]
        for p in positions:
            self.add_vertex(p)

        for p in positions:
            x, y = p
            potential_neighbors = [
                (x, y+resolution), 
                (x, y-resolution),
                (x-resolution, y),
                (x-resolution, y)
            ]
            for n in potential_neighbors:
                if n in positions:
                    edge_valid, _ = self.__edge_valid(p, n, buffer_dist=resolution)
                    if edge_valid:
                        self.add_edge(p, n)

    '''
    MA-PRM. 

    # Parameters (and their defaults)
    v = 10 # min num of nodes
    n = 35 # number of samples per iteration
    n_attempts = 150 # number of times to try sampling
    k = 9 # number of nearest neighbors to connect to
    knn_dist = 0 # min distance to connect to nearest neighbors
    sample_res = 0.2 # resolution of sampling
    sample_sparsity = 0.75 # min distance between samples.
    max_iterations = 20 # max number of iterations to run.
    '''
    def ma_prm(self, v=25, n=35, n_attempts=150, k=9, knn_dist=0, \
               sample_res=0.2, sample_sparsity=0.75, max_iterations=20):
        
        for _ in range(max_iterations):
            if self.num_connected_components() == 1 and len(self.vertices.keys()) >= v:
                break

            # Sample n points inside env. 
            samples = set()
            for _ in range(n_attempts):
                if len(samples) == n:
                    break
                # sample point
                sample = self.__sample_point(sample_res)
                sample = self.__push_sample_to_ma(sample)
                if sample == []: # invalid
                    continue
                if self.dist_to_nearest_vertex(sample) < sample_sparsity:
                    continue
                samples.add(sample)
                self.add_vertex(sample)

            for node in samples:
                has_neighbors = self.__connect_vertex_to_graph(node, k, knn_dist=knn_dist, buffer_dist=0.5)
                if not has_neighbors:
                    self.remove_vertex(node)

    ''' 
    Combination of MA-PRM and PRM.
    '''
    def both_methods(self):
        self.prm()
        ma_prm = TopologicalGraph(self.obstacles_vertices, self.environment_vertices, type="ma_prm", saveFile=False, saveImage=False)
        ma_prm_nodes, ma_prm_edges = ma_prm.get_nodes(), ma_prm.get_edges()

        added_edges = [] # edges that connect the maprm nodes to the prm nodes.

        for m_node in ma_prm_nodes:
            neighbors = self.k_nearest_neighbors(m_node, 2, min_dist=0)
            for neighbor in neighbors:
                edge_valid, _ = self.__edge_valid(m_node, neighbor, buffer_dist=0)
                if edge_valid:
                    added_edges.append((m_node, neighbor))

        self.add_many_vertices(ma_prm_nodes)
        for e in ma_prm_edges + added_edges:
            self.add_edge(e[0], e[1])

    def old(self):
        data = skeleton_to_graph(self.obstacles_vertices, self.environment_vertices)
        self.add_many_vertices(data["nodes"])
        for e in data["edges"]:
            s, t = e
            self.add_edge(s,t)  
    '''
    types:
    - prm_maprm_combined
    - ma_prm
    - prm
    - grid
    - old
    - IF NONE OF THE ABOVE: filename. 
    '''
    def calculate_topological_graph(self, type, saveFile=True, saveImage=True):
        if type == "prm_maprm_combined":
            self.both_methods()
        elif type == "ma_prm":
            self.ma_prm()
        elif type == "prm":
            self.prm()
        elif type == "grid":
            self.grid()
        elif type == "old":
            self.old()
        else:
            self.load_from_json(type)
            # raise ValueError("Invalid type of topological graph construction")

        if saveFile:
            self.save_to_json("topological_graph.json")
        
        if saveImage:
            self.visualize()

    def path_shortcut(self, path, buffer=0.2):
        if path is None:
            return None
        if len(path) == 0:
            return []
        
        for i in range(len(path)-1):
            assert self.__edge_valid(path[i], path[i+1], buffer_dist=buffer)[0], \
                f"Edge {path[i]} to {path[i+1]} is not valid with buffer={buffer}."

        new_path = [path[0]]
        start_index = 0
        end_index = 1
        while end_index < len(path):
            if self.__edge_valid(path[start_index], path[end_index], buffer_dist=buffer)[0]:
                end_index += 1
            else:
                new_path.append(path[end_index-1])
                # new_path.append((path[start_index], path[end_index-1]))
                start_index = end_index - 1
        new_path.append(path[end_index-1])
        return new_path

    def np_encoder(self, object):
        if isinstance(object, np.generic):
            return object.item()
        
    def visualize(self, ax=None):
        print("Visualizing topological graph...")
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal', adjustable='box')

        # Set the environment borders
        environment = Poly(self.environment_vertices, closed=True, edgecolor='none', facecolor='none')
        ax.add_patch(environment)

        # Set the obstacles
        for obs in self.obstacles_vertices:
            obstacles_polygon = Poly(obs,closed=True, edgecolor='black', facecolor='black')
            ax.add_patch(obstacles_polygon)

        # Set plot limits
        ax.set_xlim(self.env_bounds["xmin"], self.env_bounds["xmax"])
        ax.set_ylim(self.env_bounds["ymin"], self.env_bounds["ymax"])

        # plot the graph.
        for edge in self.get_edges():
            x, y = zip(*edge)
            ax.plot(x, y, 'o-', color='blue')
        nodes_x = [x for x, y in self.get_nodes()]
        nodes_y = [y for x, y in self.get_nodes()]
        ax.scatter(nodes_x, nodes_y, color='blue')

        plt.savefig("topological_graph.png")

# Visualize graph, env, and graph making process here, as a test scenario. 
if __name__ == "__main__":
    from matplotlib.patches import Polygon as Poly
    env_vertices = [(-6, -6), (6, -6), (6, 6), (-6, 6)]
    obstacle_vertices = [[(1, -2), (4, -2), (4, 1), (1, 1)], \
                         [(-4, -2), (-2, -2), (-2, -0), (-4, -0)], \
                         [(-4, 2), (-1, 2), (-1, 4), (-4, 4)]]

    graph = TopologicalGraph(obstacle_vertices, env_vertices, type="prm_maprm_combined")
    nodes = graph.get_nodes()
    edges = graph.get_edges()

    graph.visualize()