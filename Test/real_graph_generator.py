import numpy as np
from pygsp import graphs

positions_nodes = [
    [6.5, 4.5],
    [6.8, 5.5],
    [0.5, 7],
    [5, 6],
    [9, 7],
    [5.5, 9.0],
    [5.3, 6.5],
    [4.5, 7.5],
    [3.0, 4],
    [2, 5],
    [3.0, 3.5],
    [5.5, 2.0]
]
class RealGraphGenerator:
    def __init__(self, path_to_W):
        W = np.loadtxt(path_to_W)
        self.W = np.matrix(W)
        self.G = self.construct_real_graph()
        
    def construct_real_graph(self) : 
        # on convertit W en matrice 
        W = np.matrix(self.W)
        W[W < 0.8] = 0
        np.fill_diagonal(W, 0)
        G = graphs.Graph(W)
        G.set_coordinates(kind=positions_nodes)
        G.compute_laplacian() 
        return G