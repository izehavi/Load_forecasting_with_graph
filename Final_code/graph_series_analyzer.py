"""In this document we implement a class object that generalyzing the graph analysis with temporal series"""
import networkx as nx
from pygsp import graphs
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.signal as signal
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from AE import Autoencoder
#defintion of the class

class GraphSeriesAnalyzer : 
    """__summary__ : This class is used to analyze the graph constructed from the data of the temporal series."""
    def __init__(self, Graphe_builded, nodes_features, name_column: str, Nsize : int , AE_path ,  cutoff_freq= 1 , plot_result = False, **kwargs):
        
        self.W = Graphe_builded.W
        self.nodes_features = nodes_features
        self.name_column = name_column
        self.Nsize = Nsize
        self.plot_result = plot_result
        
        #Compute the graphe
        self.G = Graphe_builded.G
        self.L = Graphe_builded.L.toarray()
        
        # Convert the signals to models
        self.signals = self.get_signals()
        
        # Using AE 
        self.model = torch.load(AE_path)
        self.encoded_signals = self.AE()
        
        # Convert model trend+periodicity
        self.signals_filtered = self.lowpass_filter(cutoff_freq)
        self.t = np.linspace(0, self.Nsize//48, self.Nsize)# abcisse
        self.params, self.fitted_signal = self.fit_signal_with_trend_and_periodic(cutoff_freq)
        
        # Compute the FFT of the signal
        self.signals_fft, self.Lfreqs = self.signal_fft(cutoff_freq)
        
        #Compute the smoothness generalized
        self.smoothness_model = self.smoothness_calcul_generalized(self.params)
        self.smoothness_fft = self.smoothness_calcul_generalized(self.signals_fft)
        self.smoothness_AE = self.smoothness_calcul_generalized(self.encoded_signals)
        print("Done")
        
        

    def get_signals(self):
        """__summary__ : This function is used to extract the signals of the node over the time from the dataframe"""
    
        signals = [] 
        for key in self.nodes_features.keys():
            signals.append(self.nodes_features[key][self.name_column][:self.Nsize])
        signals_array = np.array(signals)
        
        #Normalization of the signal
        for i in range(signals_array.shape[0]):
            signals_array[i,:] = (signals_array[i,:] - np.mean(signals_array[i,:])) / np.std(signals_array[i,:])
        
        return signals_array
    
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
    
    def smoothness_calcul_generalized(self, params) :
        """
        Calculate the smoothness the series over the graph.
        """
        def extract_degree_matrix(L):
            """
            Extrait la matrice des degrés D à partir du Laplacien L.

            Paramètres :
            - L : scipy.sparse.csr_matrix
                Matrice de Laplacien classique (L = D - A).

            Retour :
            - D : scipy.sparse.csr_matrix
                Matrice des degrés (diagonale uniquement, zéros ailleurs).
            """
            # Extraire la diagonale de la matrice Laplacienne
            degrees = L.diagonal()
            
            # Créer une matrice diagonale sparse avec ces degrés
            D = sp.diags(degrees, format="csr")
            return D
        
        def normalize_laplacian_sym(L):
            """
            Normalise un Laplacien de manière symétrique.
            L : Matrice Laplacienne (D - A)
            D : Matrice diagonale des degrés
            """
            D = extract_degree_matrix(L)
            # Calcul de D^(-1/2)
            D_inv_sqrt = sp.diags(np.power(D.diagonal(), -0.5), format="csr")
            D_inv_sqrt[np.isinf(D_inv_sqrt.diagonal())] = 0  # Gérer les nœuds isolés
            # Normalisation symétrique
            L_sym = D_inv_sqrt @ L @ D_inv_sqrt
            return L_sym
        def smoothness_withsum(L_normalized, vec_params_normalized):
            """
            Calcule la régularité d'un signal sur un graphe.
            """
            smoothness = 0  # Initialisation de la régularité
            # Calcul with the sum
            Lsum = [] 
            for i in range (L_normalized.shape[0]):
                smoothness = 0
                for j in range (L_normalized.shape[1]):
                    smoothness -=  L_normalized[i,j] * (vec_params_normalized[i]-vec_params_normalized[j])**2
                Lsum.append(smoothness)
            return smoothness/2
        
        #Normalization of the parameters
        params_normalized = np.zeros_like(params)
        Lsmoothness = []
        for j in range(params.shape[1]):
            params_normalized[:,j] = (params[:,j] - np.mean(params[:,j])) / np.std(params[:,j])
            #print("variance",np.std(self.params[:,j]))
            #print("mean",np.mean(self.params[:,j]))
        
        # Normalisation of the Laplacian
        self.L_normalized = normalize_laplacian_sym(self.L)

        for j in range (params.shape[1]):
            Lsmoothness.append(params_normalized[:,j].T @ self.L_normalized@ params_normalized[:,j])
            
        smoothness = np.sum(np.array(Lsmoothness))
        return smoothness
    
    def lowpass_filter(self, cutoff_freq, sample_rate=48 , order=4):
        """
        Applique un filtre passe-bas à des données.

        Paramètres :
        - data : ndarray, le signal à filtrer.
        - cutoff_freq : float, fréquence de coupure (en Hz).
        - sample_rate : float, fréquence d'échantillonnage (en Hz).
        - order : int, ordre du filtre (plus grand = pente plus forte).
        - plot_result : bool, affiche le signal original et filtré si True.

        Retourne :
        - filtered_data : ndarray, le signal filtré.
        """
        plot_result = self.plot_result
        data = self.signals
        # Normalisation de la fréquence de coupure (entre 0 et 1)
        nyquist = 0.5 * sample_rate
        normalized_cutoff = cutoff_freq / nyquist

        # Conception du filtre passe-bas (Butterworth)
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
        data_filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            # Application du filtre au signal
            data_filtered[i,:] = signal.filtfilt(b, a, data[i,:])

            # Optionnel : tracer les résultats
            if plot_result:
                plt.figure(figsize=(10, 6))
                plt.plot(data[i,:], label="Signal original", alpha=0.7)
                plt.plot(data_filtered[i,:], label=f"Signal filtré (fc={cutoff_freq} Hz)", color='red')
                plt.legend()
                plt.xlabel("Temps (échantillons)")
                plt.ylabel("Amplitude")
                plt.title("Filtrage passe-bas")
                plt.show()

        return data_filtered


    def fit_signal_with_trend_and_periodic(self, cutoff_freq = 1 , poly_degree=4, num_periodic=4):
        """
        Ajuste un modèle combinant une tendance polynomiale (degré variable)
        et plusieurs fonctions périodiques avec cosinus (avec déphasage) à un signal donné.

        Paramètres :
        - t : ndarray, les points temporels.
        - signal : ndarray, le signal à ajuster.
        - poly_degree : int, degré du polynôme pour la tendance.
        - num_periodic : int, nombre de composantes périodiques.
        - plot_result : bool, affiche le graphique des résultats si True.

        Retour :
        - params : ndarray, paramètres ajustés du modèle.
        - fitted_signal : ndarray, signal ajusté.
        """
        # Définition dynamique du modèle avec polynôme et cosinus avec déphasage
        def model(t, *params):
            # Extraction des coefficients
            poly_params = params[:poly_degree + 1]  # Coefficients du polynôme
            periodic_params = params[poly_degree + 1:]  # Coefficients des composantes périodiques
            
            # Tendance polynomiale
            trend = sum(poly_params[i] * t**i for i in range(poly_degree + 1))
            
            # Fonction périodique avec un nombre variable de termes
            periodic = 0
            for i in range(num_periodic):
                amplitude = periodic_params[3 * i]
                freq = periodic_params[3 * i + 1]
                phase = periodic_params[3 * i + 2]
                periodic += amplitude * np.cos(2 * np.pi * freq * t + phase)
            
            return trend + periodic
        plot_result = self.plot_result
        fitted_signal = np.zeros_like(self.signals_filtered)
        t = self.t
        Lparams = []
        for i in range(self.signals_filtered.shape[0]):
            # Définition des paramètres initiaux
            p0 = [0] * (poly_degree + 1)
            
            
            for j in range(num_periodic):
                p0 += [1, cutoff_freq/num_periodic*(j+1), 0]  # Amplitude initiale, fréquence initiale, phase initiale

        
            # Ajustement des paramètres avec curve_fit
            params, _ = curve_fit(model, t, self.signals_filtered[i,:], p0=p0, maxfev=10000)
            Lparams.append(params)
            # Signal ajusté
            fitted_signal[i,:] = model(t, *params)

            # Optionnel : afficher les résultats
            if plot_result:
                plt.figure(figsize=(10, 6))
                plt.plot(t, self.signals_filtered[i,:], label="Signal original", alpha=0.7)
                plt.plot(t, fitted_signal[i,:], label="Signal ajusté (modèle)", linestyle="--", color="red")
                plt.legend()
                plt.xlabel("Temps")
                plt.ylabel("Amplitude")
                plt.title(f"Ajustement : Tendance polynomiale (degré {poly_degree}) + {num_periodic} composantes périodiques (cosinus)")
                plt.show()

        return np.array(Lparams), fitted_signal
    
    def signal_fft(self, cutoff_freq):
        """
        Compute the FFT of the signal
        """
        N = self.Nsize
        T = 1.0 / 48.0
        yf = np.fft.fft(self.signals_filtered)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        
        # Selection of values under the cutoff frequency
        xf_cut = xf[xf <= cutoff_freq]
        yf_cut = yf[:, :xf_cut.shape[0]]
        
        if self.plot_result:
            plt.figure(figsize=(10, 6))
            for i in range(yf.shape[0]):
                plt.plot(xf_cut, 2.0/N * np.abs(yf_cut[i,:]))
            plt.grid()
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("Amplitude")
            plt.title("Transformée de Fourier du signal")
            plt.show()
        return np.abs(yf_cut), xf_cut
    
    def AE (self) : 
        """
        Load the Autoencoder model and compute the reconstruction error
        """
        def MAPE(y_true, y_pred):
            """
            Calcule le MAPE (Mean Absolute Percentage Error) entre les valeurs réelles et prédites.

            Arguments :
            - y_true (torch.Tensor) : Valeurs réelles (taille N)
            - y_pred (torch.Tensor) : Valeurs prédites (taille N)
            
            Retour :
            - mape (torch.Tensor) : Erreur MAPE (en pourcentage)
            """
            epsilon = 1e-10  # Pour éviter la division par zéro
            y_true = y_true.clamp(min=epsilon)  # Évitez la division par zéro
            mape = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
            return mape
        model = self.model
        encoded_signal = model.encode(torch.tensor(self.signals))
        
        
        criterion = nn.MSELoss()
        signals = torch.tensor(self.signals)
        outputs = model.predict(signals)
        loss = criterion(outputs, signals)
        print("RMSE", loss.item())
        
        mape_value  = MAPE(signals, outputs)
        print("MAPE", mape_value)
        
        return encoded_signal.detach().numpy()