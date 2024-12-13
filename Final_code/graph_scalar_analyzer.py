import networkx as nx
from pygsp import graphs
import numpy as np
import matplotlib.pyplot as plt


class GraphScalarAnalyzer:
    """
    Class to analyze graphs.
    
    kwargs : position of the nodes.
    Args : 
        W (np.ndarray): Weight matrix of the graph.
        name_column (str): Name of the column containing the signal of interest.
        row (int): Index of the row containing the value of the signal at a given node.
        
    """
    def __init__(self, Graph_builded , nodes_features, name_column: str, row : int, **kwargs):
        self.W = Graph_builded.W
        self.nodes_features = nodes_features
        self.signal = self.get_signal(name_column, row)
        self.G = Graph_builded.G
        self.L = Graph_builded.L
        
        
        self.smoothness = self.smoothness_calcul()
        self.U, self.lambdas, self.GFT_signal = self.graph_fourier_transform() # to have the U and lambdas of the GFT (Eigeenvalues and eigenvectors)
        self.bandwith_general = self.bandwith_generalized()
        # self.energyE = self.energy_distribution_calcul()
        # self.energyG = self.energy_distribution_interference_reduced()
        # self.bandwithE = self.bandwith(self.energyE)#todo bandwith negative > à revoir
        # self.bandwithG = self.bandwith(self.energyG)
        # self.kindexE = self.find_mainfrequency_ofnode(self.energyE)
        #self.kindexG = self.find_mainfrequency_ofnode(self.energyG)
        
    def smoothness_calcul(self):
        """Calcul the smoothness of a signal on a graph.
        Args:
            self.G (Graph): Graph on which the signal is defined.
            self.signal (np.ndarray): Signal on the graph.
        Returns:
            float: Smoothness of the signal.
        """
        return self.signal.T @ self.G.L @ self.signal/len(self.signal)
         
    
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
    
    def bandwith_generalized(self):
        """
        Computes the generalized bandwidth of the signal on the graph.
        
        Args:
            G (np.ndarray): Vertex-frequency energy distribution matrix of size (N x N),
                            where N is the number of vertices (or eigenvalues).
            lambdas (np.ndarray): Array of eigenvalues λ_k of size N.

        Returns:
            float: Generalized bandwidth of the signal.
        """
        lambdas = self.lambdas
        
        # Calculate the average frequency of the signal
        lambda_avg = np.sum(lambdas * np.abs(self.GFT_signal)) / np.sum(np.abs(self.GFT_signal))
        
        # Compute the numerator and denominator for the bandwidth  
        numerator = np.sum((lambdas - lambda_avg)**2 * self.GFT_signal**2)
        denominator = np.sum(self.GFT_signal**2)
        
        # Compute the bandwidth
        bandwidth = np.sqrt(numerator / denominator) if denominator != 0 else 0
        
        return bandwidth
    
    def energy_distribution_calcul(self):
        """_summary_
        Calculate the energy distribution of the signal on the graph for each node and each frequency.
        
        returns:
            np.ndarray: Energy distribution of the signal on the graph E(n,k).
        """
        
        def plot_energy(energy) :
            plt.figure(figsize=(8, 6))
            plt.imshow(energy, cmap='viridis', interpolation='nearest', aspect='auto')
            plt.colorbar(label='Énergie')
            plt.title("Distribution de l'énergie du signal sur le graphe")
            plt.xlabel("Fréquences (indices des valeurs propres)")
            plt.ylabel("Nœuds")
            plt.legend()
            plt.show()
        
        energy = np.zeros((self.U.shape[0], self.U.shape[1]))
        for n in range(self.U.shape[0]):  # Parcourir les nœuds
            for k in range(self.U.shape[1]):  # Parcourir les fréquences
                energy[n, k] = self.signal[n] * self.U[n, k] * self.GFT_signal[k]

        # Visualisation
        plot_energy(energy)

        return energy
    
    def energy_distribution_interference_reduced(self):
        """_summary_
        Calculate the energy distribution of the signal on the graph for each node and each frequency with a Kernel function to reduce interferencies.
        
        returns:
            np.ndarray: Energy distribution of the signal on the graph with less interferencies G(n,k).
        """
        def kernel(p, k, q, alpha = 1):
            """_summary_
            Computes the value of φ(p, k, q) based on the given formula.

            Args:
                lambdas (list or np.ndarray): List of eigenvalues (λ values).
                alpha (float): Scaling factor in the exponential.
                p, k, q (int): Indices for which φ(p, k, q) is computed.

            Returns:
                float: Value of φ(p, k, q).
            """
            lambdas = self.lambdas
            N = len(lambdas)
            
            # Avoid division by zero for q == p
            if q == p:
                return 1.0 if k == p else 0.0
            
            # Compute |λp - λk| / |λp - λq|
            numerator = np.exp(-alpha * np.abs(lambdas[p] - lambdas[k]) / np.abs(lambdas[p] - lambdas[q]))
            
            # Compute s(q, p)
            denominator = sum(
                np.exp(-alpha * np.abs(lambdas[p] - lambdas[j]) / np.abs(lambdas[p] - lambdas[q]))
                for j in range(N)
            )
            
            # Compute φ(p, k, q)
            phi = numerator / denominator
            return phi

        def plot_vertex_frequency_distribution(G_matrix):
            """
            Plots the vertex-frequency energy distribution (G(n, k)) and its marginal distributions.
            
            Args:
                G_matrix (np.ndarray): 2D array of energy values G(n, k) where n represents vertices
                                    and k represents frequencies.
            """
            # Compute marginal distributions
            vertex_marginals = np.sum(G_matrix, axis=1)  # Sum over frequencies (rows)
            frequency_marginals = np.sum(G_matrix, axis=0)  # Sum over vertices (columns)

            # Create figure and gridspec layout
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(2, 3, width_ratios=[4, 0.1, 1], height_ratios=[4, 1], hspace=0.3, wspace=0.3)

            # Main heatmap plot (G(n, k))
            ax_main = fig.add_subplot(gs[0, 0])  # Heatmap
            im = ax_main.imshow(G_matrix, aspect='auto', cmap='hot', interpolation='nearest')
            ax_main.set_title("Vertex-frequency energy distribution")
            ax_main.set_xlabel("Frequency index")
            ax_main.set_ylabel("Vertex index")

            # Colorbar
            cbar_ax = fig.add_subplot(gs[0, 1])  # Cellule pour la colorbar
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', label="Energy")

            # Marginal distribution over frequencies
            ax_right = fig.add_subplot(gs[0, 2], sharey=ax_main)  # Marginal distribution on the right
            ax_right.plot(frequency_marginals, range(len(frequency_marginals)), 'o-')
            ax_right.set_xlabel("Marginals")
            ax_right.set_ylabel("Spectral index")
            ax_right.invert_yaxis()  # Align y-axis with the heatmap

            # Marginal distribution over vertices (aligned below heatmap)
            ax_bottom = fig.add_subplot(gs[1, 0])  # Aligns exactly below the heatmap
            ax_bottom.stem(range(len(vertex_marginals)), vertex_marginals, basefmt=" ")
            ax_bottom.set_xlabel("Vertex index")
            ax_bottom.set_ylabel("Marginals")
            ax_bottom.yaxis.set_label_coords(-0.1, 0.5)  # Adjust y-axis label position

            # Hide the bottom-right corner (empty space)
            ax_empty = fig.add_subplot(gs[1, 1:])
            ax_empty.axis('off')  # Disable this subplot completely

            # Show the full plot
            plt.show()


        def compute_G(n, k, alpha):
            """
            Computes the vertex-frequency energy distribution G(n, k).

            Args:
                U (np.ndarray): Matrix of eigenvectors (columns are eigenvectors of the Laplacian, size NxN).
                X (np.ndarray): Signal in the spectral domain (size N).
                kernel (function): Function to compute phi(p, k, q) given p, k, q.
                n (int): Vertex index.
                k (int): Frequency index.
                alpha (float): Scaling parameter used in phi(p, k, q).

            Returns:
                float: Value of G(n, k).
            """
            U=self.U
            X=self.GFT_signal
            
            N = U.shape[0]  # Number of vertices / frequencies
            G_nk = 0.0

            # Double summation over p and q
            for p in range(N):
                for q in range(N):
                    # Compute terms of the summation
                    X_p = X[p]                  # X(p)
                    X_q_star = np.conj(X[q])    # X*(q)
                    u_p_n = U[n, p]             # u_p(n)
                    u_q_n_star = np.conj(U[n, q])  # u_q*(n)
                    phi_p_k_q = kernel(p, k, q, alpha)  # phi(p, k, q)

                    # Add the term to the summation
                    G_nk += X_p * X_q_star * u_p_n * u_q_n_star * phi_p_k_q

            return G_nk
        
        energyG = np.zeros((self.U.shape[0], self.U.shape[1]))
        
        
        for n in range(self.U.shape[0]):
            for k in range(self.U.shape[1]):
                energyG[n, k] = compute_G(n, k, alpha=1)
        
        #Visualization
        plot_vertex_frequency_distribution(energyG)
        
        return energyG
    
    def bandwith(self, G):
        """
        the local smoothness bandwith : Computes the variance of the spectral energy distribution σ²_λ(n) for each vertex n.

        Args:
            G (np.ndarray): Vertex-frequency energy distribution matrix of size (N x N),
                            where N is the number of vertices (or eigenvalues).
            lambdas (np.ndarray): Array of eigenvalues λ_k of size N.

        Returns:
            np.ndarray: Array of σ²_λ(n) values, one for each vertex.
        """
        lambdas = self.lambdas
        N = G.shape[0]  # Number of vertices or frequencies
        sigma_lambda_squared = np.zeros(N)

        for n in range(N):
            # Compute λ(n) as the weighted mean of eigenvalues for vertex n
            lambda_n = np.sum(lambdas * G[n, :]) / np.sum(G[n, :])

            # Compute the numerator and denominator for σ²_λ(n)
            numerator = np.sum((lambdas - lambda_n)**2 * G[n, :])
            denominator = np.sum(G[n, :])

            # Compute σ²_λ(n) for vertex n
            sigma_lambda_squared[n] = numerator / denominator if denominator != 0 else 0

        return sigma_lambda_squared
        
    def find_mainfrequency_ofnode(self, G):
        """
        Find the main frequency of the signal.
        
        Args:
            G (np.ndarray): Energy distribution of the signal on the graph.
        
        Returns:
            List: List of Index of the main frequency for each node.
        """
        kindex =[]
        for n in range(G.shape[0]):
            kindex.append(np.argmax(G[n, :]))
        
        return kindex   