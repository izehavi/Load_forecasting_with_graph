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
        return self.W
    
    def keep_top_n(self, matrix: np.ndarray, N: int) -> np.ndarray:
        """ 
        Keep the N largest values in a matrix and set the others to zero.
        
        Args:
            matrix (np.ndarray): Input matrix.
            N (int): Number of largest values to keep.
        returns:
            np.ndarray: Filtered matrix with only the N largest values.
        """
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
    
    



