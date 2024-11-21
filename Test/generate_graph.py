import numpy as np
from pygsp import graphs

def generate_random_graph(n_nodes=15, sparsity=0.8):
    """Génère un graphe aléatoire."""
    rs = np.random.RandomState(42)
    W = rs.uniform(size=(n_nodes, n_nodes))
    W[W < sparsity] = 0
    W = W + W.T
    np.fill_diagonal(W, 0)
    G = graphs.Graph(W)
    G.set_coordinates('spring')
    G.compute_laplacian() 
    return G

def generate_ring_graph(n_nodes=15):
    """Génère un graphe en anneau."""
    G = graphs.Ring(N=n_nodes)
    return G

    
    