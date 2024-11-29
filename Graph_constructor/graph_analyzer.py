import networkx as nx
from pygsp import graphs
import numpy as np
import matplotlib.pyplot as plt


class GraphAnalyzer:
    """
    Class to analyze graphs.
    
    kwargs : position of the nodes.
    Args : 
        W (np.ndarray): Weight matrix of the graph.
        name_column (str): Name of the column containing the signal of interest.
        row (int): Index of the row containing the value of the signal at a given node.
        
    """
    def __init__(self, W, nodes_features, name_column: str, row : int , **kwargs):
        self.W = W
        self.nodes_features = nodes_features
        self.signal = self.get_signal(name_column, row)
        self.G = self.construct_real_graph(**kwargs)
        self.L = self.G.L
        self.smoothness = self.smoothness_calcul()
        self.U, self.lambdas, self.GFT_signal = self.graph_fourier_transform() # to have the U and lambdas of the GFT (Eigeenvalues and eigenvectors)
        self.energy = self.energy_distribution_calcul()
        
    def smoothness_calcul(self):
        """Calcul the smoothness of a signal on a graph.
        Args:
            self.G (Graph): Graph on which the signal is defined.
            self.signal (np.ndarray): Signal on the graph.
        Returns:
            float: Smoothness of the signal.
        """
        return self.signal.T @ self.G.L @ self.signal
    
    
        
    def construct_real_graph(self, **kwargs) :
        """
        Construct a real graph from the weight matrix W.
        """ 
        
        # construction of positions_nodes with the latitude and longitude of the cities
        latitude = kwargs.get("LATITUDE", None)
        Longitude = kwargs.get("LONGITUDE", None)
        positions_nodes = []
        for i in range(len(latitude)):
            positions_nodes.append([latitude[i], Longitude[i]])
            
        G = graphs.Graph(self.W)
        G.set_coordinates(kind=positions_nodes)# TODO mettre les positions des villes
        G.compute_laplacian() 
        
        return G
    
    def get_signal(self, name_column : str, row : int): 
        """
        Get the signal of interest to generate the signal graph.
        
        Args:
            self.nodes_features (dict): Dictionary containing the features of the nodes.
            name_column (str): Name of the column containing the signal of interest.
            row (int): Index of the row containing the value of the signal at a given node.
        
        Returns:
            np.ndarray: Vector of the signal.
        """
        
        # calculating the shape of the matrix signals 
        signal = []
        
        for key in self.nodes_features.keys():
            signal.append(self.nodes_features[key][name_column][row])
        signal_array = np.array(signal)
        
        #Normalization of the signal
        signal_array = (signal_array - np.mean(signal_array)) / np.std(signal_array)
        return signal_array
    
    def graph_fourier_transform(self):
        """_summary_
        Calculate the Fourier Transform of the signal on the graph.
        
        Returns:
            np.ndarray: Fourier Transform of the signal.
        """
        self.G.compute_fourier_basis()
        self.U = self.G.U
        self.lambdas = self.G.e
        self.GFT_signal = self.U.T @ self.signal
        return self.U, self.lambdas, self.GFT_signal
    
    
    def energy_distribution_calcul(self):
        """Calcule et visualise la distribution d'énergie du signal."""
        energy = np.zeros((self.U.shape[0], self.U.shape[1]))
        for n in range(self.U.shape[0]):  # Parcourir les nœuds
            for k in range(self.U.shape[1]):  # Parcourir les fréquences
                energy[n, k] = self.signal[n] * self.U[n, k] * self.GFT_signal[k]

        # Visualisation
        plt.figure(figsize=(8, 6))
        plt.imshow(energy, cmap='viridis', interpolation='nearest', aspect='auto')
        plt.colorbar(label='Énergie')
        plt.title("Distribution de l'énergie du signal sur le graphe")
        plt.xlabel("Fréquences (indices des valeurs propres)")
        plt.ylabel("Nœuds")
        plt.show()

        return energy
        