import numpy as np
import pandas as pd
from pygsp import graphs




class GraphBuilder:
    """
    Class to build a graph from data already calculated by another code.
    """

    def __init__(self,  L_path_to_W,  **kwargs):
        """
        Initializes the graph builder with data and parameters.

        Args:
            L_Ws (list): List of Edges matrices already calculated from another code.
            df_pos (pd.DataFrame): DataFrame containing the positions of the nodes.
            num_nodes (int): Total number of nodes.
            kwargs (dict): Optional parameters.
        """

        self.kwargs = kwargs
        self.W_basic = self.load_Wmatrix(L_path_to_W)
        self.W = self.keep_top_n(20)
        self.G = self.construct_real_graph(**kwargs)
        self.L = self.G.L


        
    def load_Wmatrix(self, path_to_W):
        W = np.loadtxt(path_to_W)
        self.W = np.matrix(W)
        np.fill_diagonal(self.W, 0)
        return self.W

    def construct_real_graph(self, **kwargs) :
        """
        Construct a real graph from the weight matrix W.
        """ 
        def normalize_adjacency_matrix(A):
            """
            Normalise la matrice d'adjacence A.
            
            Arguments :
            - A (numpy.ndarray) : Matrice d'adjacence (carrée) de taille (N, N)
            
            Retour :
            - A_norm (numpy.ndarray) : Matrice d'adjacence normalisée de taille (N, N)
            """
            # 1. Calcul du degré de chaque nœud (somme des lignes de la matrice A)
            degrees = np.sum(A, axis=1)  # Degré de chaque nœud (somme des connexions de chaque nœud)
            
            # 2. Évitez la division par zéro (pour les nœuds isolés)
            degrees[degrees == 0] = 1  # Pour éviter la division par zéro
            
            # 3. Calcul de D^(-1/2)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
            # 4. Normalisation symétrique : A_hat = D^(-1/2) * A * D^(-1/2)
            A_norm = D_inv_sqrt @ A @ D_inv_sqrt  # Multiplication matricielle
            return A_norm
        
        # construction of positions_nodes with the latitude and longitude of the cities
        latitude = kwargs.get("LATITUDE", None)
        Longitude = kwargs.get("LONGITUDE", None)
        positions_nodes = []
        for i in range(len(latitude)):
            positions_nodes.append([latitude[i], Longitude[i]])
        
        # construction of the graph
        self.W = normalize_adjacency_matrix(self.W)
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






