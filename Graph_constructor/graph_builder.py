import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
import pandas as pd



class GraphBuilder:
    """
    Class to build a graph from data already calculated by another code.
    """

    def __init__(self, Nodes_features,  L_path_to_W = None,  **kwargs):
        """
        Initializes the graph builder with data and parameters.

        Args:
            L_Ws (list): List of Edges matrices already calculated from another code.
            df_pos (pd.DataFrame): DataFrame containing the positions of the nodes.
            num_nodes (int): Total number of nodes.
            kwargs (dict): Optional parameters.
        """

        self.kwargs = kwargs
        self.Nodes_features = Nodes_features

    
        
    def load_Wmatrix(self, path_to_W):
        W = np.loadtxt(path_to_W)
        self.W = np.matrix(W)
        np.fill_diagonal(self.W, 0)
        return self.W
    
    def keep_top_n(self, N: int) -> np.ndarray:
        """ 
        Keep the N largest values in a matrix and set the others to zero.
        
        Args:
            matrix (np.ndarray): Input matrix.
            N (int): Number of largest values to keep.
        returns:
            np.ndarray: Filtered matrix with only the N largest values.
        """
        matrix = self.W
        # Flatten the matrix to find the N largest values
        flat_vector = matrix.flatten()
        flat_matrix = np.array(flat_vector).flatten()
        
        # If N exceeds the total size of the matrix, keep all elements
        if N >= flat_matrix.shape[0]:
            return matrix
        
        # Find the Nth largest value
        sorted = np.sort(flat_matrix)
        threshold = sorted[-N]
        
        # Create a filtered matrix where only values >= threshold are kept
        filtered_matrix = np.where(matrix >= threshold, matrix, 0)
        
        self.W_filtered = filtered_matrix
        return filtered_matrix
    

    def filter_edges_by_energy(self, s):
        """
        Filtre les arêtes d'un graphe pour conserver celles représentant au moins s% de l'énergie totale.

        Args:
            adj_matrix (np.ndarray): Matrice d'adjacence du graphe (symétrique pour un graphe non orienté).
            s (float): Pourcentage d'énergie à conserver (entre 0 et 1).

        Returns:
            filtered_matrix (np.ndarray): Matrice d'adjacence filtrée avec uniquement les arêtes sélectionnées.
            total_energy (float): Énergie totale calculée avant filtrage.
            selected_energy (float): Énergie cumulée des arêtes retenues.
        """
        
        # adj_matrix
        adj_matrix = self.W
        # Calculer l'énergie totale (racine carrée de la somme des carrés des poids)
        total_energy = np.sum(adj_matrix ** 2)
        #total_energy = np.sum(adj_matrix)
        # Calculer l'énergie relative pour chaque lien (poids / énergie totale)
        matrix = adj_matrix**2 / total_energy
        
        flat_vector = matrix.flatten()
        flat_matrix = np.array(flat_vector).flatten()
        
        
        # Find the Nth largest value
        sorted = np.sort(flat_matrix)

        # on cherche le seuil correspondant à s% d'énergie
        i=0
        if s<1 :
            cumulative_energy = 0
            while cumulative_energy <= s :
                cumulative_energy += sorted[-i]
                i+=1
            threshold = sorted[-i]
        print("cumulative_energy", cumulative_energy)
        # Create a filtered matrix where only values >= threshold are kept
        filtered_matrix = np.where(matrix >= threshold, self.W, 0)
        
        self.W_filtered = filtered_matrix

        
        
        return self.W_filtered

    
    



