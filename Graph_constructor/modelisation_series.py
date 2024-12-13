#%%
""" In this document we will test the functions of the folder (dataloader.py, graph_builder.py)"""
%load_ext autoreload
%autoreload 2

 # Initialisation et importation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader import DataLoader
from graph_builder import GraphBuilder
from visualizations import GraphPlotter
#from graph_analyzer import GraphAnalyzer

path_train = r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\Graph_constructor\train.csv"
path_test = r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\Graph_constructor\test.csv"

L_path_to_W = [
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\examples_graph\correlation\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\examples_graph\distsplines\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\examples_graph\distsplines2\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\examples_graph\dtw\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\examples_graph\gl3sr\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\examples_graph\precision\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\examples_graph\space\W.txt"
]
path_test = r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\examples_graph\correlation\W.txt"

df_pos = pd.DataFrame(
    {'VILLE': ['LILLE', 'ROUEN', 'PARIS', 'STRASBOURG', 'BREST', 'NANTES', 'ORLEANS', 'DIJON', 'BORDEAUX', 'LYON',
              'TOULOUSE', 'MARSEILLE'],
    'LATITUDE': [50.6365654, 49.4404591, 48.862725, 48.584614, 
                 48.3905283, 47.2186371, 47.9027336, 47.3215806, 
                 44.841225, 45.7578137, 43.6044622, 43.2961743],
    'LONGITUDE': [3.0635282, 1.0939658, 2.287592, 7.7507127, 
                  -4.4860088, -1.5541362, 1.9086066, 5.0414701, 
                  -0.5800364, 4.8320114, 1.4442469, 5.3699525],
    'REGION': ['Hauts_de_France', 'Normandie', 'Ile_de_France', 'Grand_Est', 'Bretagne', 'Pays_de_la_Loire', 
                'Centre_Val_de_Loire', 'Bourgogne_Franche_Comte', 'Nouvelle_Aquitaine', 'Auvergne_Rhone_Alpes', 
                'Occitanie', 'Provence_Alpes_Cote_d_Azur'],
    'SUPERFICIE_REGION': [31813, 29906, 12011, 57433, 27208, 32082, 39151, 47784, 83809, 69711, 72724, 31400],
    'POPULATION_REGION': [5987172, 3307286, 12395148, 5542094, 3402932, 3873096, 2564915, 2785393, 6081985,
                          8153233, 6053548, 5131187]
    })
data = DataLoader(path_train, path_test, kwargs={"start_date": "2018-01-01", "end_date": None})



# Test the GraphBuilder class
Graph = GraphBuilder(L_path_to_W)
dictW = {}
for i in range(len(L_path_to_W)):
    W = Graph.load_Wmatrix(L_path_to_W[i])
    print(W)
    dictW["W"+str(i)] = W

    # We are testing different filter of edges 
    #W_threshold = Graph.keep_top_n(60)# keep the 60% of the edges

    W_threshold = Graph.filter_edges_by_energy(0.8) # 80% of the energy
    print(W_threshold)
    # Test the GraphPlotter class
    plotter = GraphPlotter(W_threshold, df_pos)
    #plotter.plot_graph(title="Graph", figsize=(10, 8))
    #plotter.plot_graph_on_map(title="Graph on Map")



# We are ploting the temperature over the time for a given node
# calculating the shape of the matrix signals 
def get_signals(nodes_features, name_column, N :int):
    """
    Get the signals of interest to generate the signal graph.
    
    Args:
        self.nodes_features (dict): Dictionary containing the features of the nodes.
    
    Returns:
        np.ndarray: Matrix of the signals.
    """
    signals = [] #np.zeros((Nbnodes, N))
    for key in nodes_features.keys():
        signals.append(nodes_features[key][name_column][:N])
    signals_array = np.array(signals)
    
    #Normalization of the signal
    for i in range(signals_array.shape[0]):
        signals_array[i,:] = (signals_array[i,:] - np.mean(signals_array[i,:])) / np.std(signals_array[i,:])
    return np.array(signals_array)

    
def plot_signals(signals) : 
    """ In this function wi will plot the signals for each nodes over the time on a different plot"""
    plt.figure(figsize=(15, 10))
    for i in range(signals.shape[0]):
        plt.plot(signals[i,:], label=f"Node {i}")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title("Temperature of each node over the time")
    plt.legend()
    
    
    
signals = get_signals(data.nodes_dataframe, "temp", 1000)
plot_signals(signals)


# %% (commented) Test the GraphAnalyzer class only on one W matrix and on Temperature
from graph_analyzer import GraphAnalyzer


def plot_everything(W, data, df_pos, key):
    """__summary__ : This function will plot the evolution of many features over the time : 
    - The energy of the temperature on the graph over the time
    - The smoothness of the temperature on the graph over the time
    - The main frequency of each node over the time
    - The bandwith of the graph of each node over the time
    - The eigeenvalues of the graph
    - GFT of the graphe over the time
    -"""
    # Création d'une figure
    N = 100
    Lsmoothness = []
    Lmain_freq = np.zeros((W.shape[0],N)) # We will store the main frequency of each node over the time
    Lbandwith = np.zeros((W.shape[0],N)) # We will store the bandwith of each node over the time
    eigeenvalues = np.zeros(W.shape[0]) # We will store the eigeenvalues of the graph
    Lgft = np.zeros((W.shape[0],N)) # We will store the GFT of the graph over the time
    L_img_energy = np.zeros((N, W.shape[0],W.shape[1]))
    
    for i in tqdm(range(N)) :
        analyzed_graph = GraphAnalyzer(W, data.nodes_dataframe, key, i, **df_pos)
        Lsmoothness.append(analyzed_graph.smoothness)
        Lmain_freq[:,i] = analyzed_graph.kindexE
        Lbandwith[:,i] = analyzed_graph.bandwithE
        Lgft[:,i] = analyzed_graph.GFT_signal
        L_img_energy[i, :, :] = analyzed_graph.energyE
    
    eigeenvalues = analyzed_graph.lambdas
    
    
    # 1. Plotting the smoothness
    plt.plot(Lsmoothness)
    plt.xlabel("Time")
    plt.ylabel("Smoothness")
    plt.title("Smoothness of the temperature over the time")
    plt.legend()
    plt.show()
    
    
    
    # 2. Plotting the main frequency of each node over the time 
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 12))  # 6 lignes, 2 colonnes
    axes = axes.flatten()
    for i in range(W.shape[0]):
        axes[i].plot(Lmain_freq[i,:])  # Trace le signal
        axes[i].set_title(f'main frequency of node {i} over the time')  # Titre
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('main frequency')
    plt.tight_layout()
    plt.show()
    
    
    # 3. Plotting the bandwith of each node over the time
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 12))  # 6 lignes, 2 colonnes
    axes = axes.flatten()
    for i in range(W.shape[0]):
        axes[i].plot(Lbandwith[i,:])  # Trace le signal
        axes[i].set_title(f'bandwith of node {i} over the time')  # Titre
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('main frequency')
    plt.tight_layout()
    plt.show()
    
    # 4. Plotting the eigeenvalues of the graph
    plt.clf()
    plt.plot(eigeenvalues)
    plt.xlabel("Node")
    plt.ylabel("Eigeenvalues")
    plt.title("Eigeenvalues of the graph")
    plt.legend()
    plt.show()
    
    # 5. Plotting the GFT of the graph over the time
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 12))  # 6 lignes, 2 colonnes
    axes = axes.flatten()
    for i in range(W.shape[0]):
        axes[i].plot(Lgft[i,:])  # Trace le signal
        axes[i].set_title(f'Magnitude of the signal at the frequency {i} over the time')  # Titre
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Magnitude')
    plt.tight_layout()
    plt.show()
    
    plt.ion()
    fig, ax = plt.subplots()
    img_Energy = L_img_energy[0, :, :]
    img = ax.imshow(img_Energy, cmap='viridis', interpolation='nearest')

    for i in tqdm(range(N)) :
        img.set_data(L_img_energy[i, :, :])  # Mettre à jour l'image
        plt.draw()
        plt.pause(0.2)

    plt.ioff()
    plt.show()


#plot_everything(dictW["W3"], data, df_pos, "temp")


        



# %% Test 1 to fit temporal series
from scipy.optimize import curve_fit

def fit_trend_and_periodic(t, signal, plot_result=True):
    """
    Ajuste un modèle combinant une tendance polynomiale (ordre 3) 
    et une fonction périodique à un signal donné.

    Paramètres :
    - t : ndarray, les points temporels.
    - signal : ndarray, le signal à ajuster.
    - plot_result : bool, affiche le graphique des résultats si True.

    Retour :
    - params : ndarray, paramètres ajustés du modèle.
    - fitted_signal : ndarray, signal ajusté.
    """
    # Définition du modèle
    def model(t, a0, a1, a2, a3, b1, b2, b3, b4,b5,b6, c1, c2, c3):
        trend = a0 + a1 * t + a2 * t**2 + a3 * t**3
        periodic = (b1 * np.sin(c1 * t) + b2 * np.cos(c1 * t) + b3 * np.sin(c2 * t) + b4*np.cos(c2 * t) +b5 * np.sin(c3 * t) + b6*np.cos(c3 * t))
        return trend + periodic

    # Ajustement des paramètres avec des valeurs initiales
    p0 = [0, 0, 0, 0, 1, 1, 1,1,1, 1, 2 * np.pi, 10 * np.pi , 100*np.pi]
    params, _ = curve_fit(model, t, signal, p0=p0)

    # Signal ajusté
    fitted_signal = model(t, *params)

    # Optionnel : afficher les résultats
    if plot_result:
        plt.figure(figsize=(10, 6))
        plt.plot(t, signal, label="Signal original", alpha=0.7)
        plt.plot(t, fitted_signal, label="Signal ajusté (modèle)", linestyle="--", color="red")
        plt.legend()
        plt.xlabel("Temps")
        plt.ylabel("Amplitude")
        plt.title("Ajustement : Tendance polynomiale + Fonction périodique")
        plt.show()

    return params, fitted_signal

# Ajustement du signal
params, fitted_signal = fit_trend_and_periodic(np.arange(1000), signals[0,:], plot_result=True)
print(params)
# %%
from scipy.optimize import curve_fit

def fit_signal_with_trend_and_periodic(t, signal, poly_degree=3, num_periodic=2, plot_result=True):
    """
    Ajuste un modèle combinant une tendance polynomiale (degré variable)
    et plusieurs fonctions périodiques (fréquences libres) à un signal donné.

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
    # Définition dynamique du modèle avec polynôme et fonctions périodiques
    def model(t, *params):
        # Extraction des coefficients
        poly_params = params[:poly_degree + 1]  # Coefficients du polynôme
        periodic_params = params[poly_degree + 1:]  # Coefficients des composantes périodiques
        
        # Tendance polynomiale
        trend = sum(poly_params[i] * t**i for i in range(poly_degree + 1))
        
        # Fonction périodique avec un nombre variable de termes
        periodic = 0
        for i in range(num_periodic):
            b_sin = periodic_params[3 * i]
            b_cos = periodic_params[3 * i + 1]
            freq = periodic_params[3 * i + 2]
            periodic += b_sin * np.sin(2 * np.pi * freq * t) + b_cos * np.cos(2 * np.pi * freq * t)
        
        return trend + periodic

    # Définition des paramètres initiaux
    p0 = [0] * (poly_degree + 1)
    for i in range(num_periodic):
         p0 += [2, 2, 1*(i*2)] * num_periodic  # (poly_degree + 1) pour le polynôme + 3 par périodique

    # Ajustement des paramètres avec curve_fit
    params, _ = curve_fit(model, t, signal, p0=p0, maxfev=100000)

    # Signal ajusté
    fitted_signal = model(t, *params)

    # Optionnel : afficher les résultats
    if plot_result:
        plt.figure(figsize=(10, 6))
        plt.plot(t, signal, label="Signal original", alpha=0.7)
        plt.plot(t, fitted_signal, label="Signal ajusté (modèle)", linestyle="--", color="red")
        plt.legend()
        plt.xlabel("Temps")
        plt.ylabel("Amplitude")
        plt.title(f"Ajustement : Tendance polynomiale (degré {poly_degree}) + {num_periodic} composantes périodiques")
        plt.show()

    return params, fitted_signal

# Exemple d'utilisation
if __name__ == "__main__":
    # Génération d'un signal synthétique
    t = np.linspace(0, 1000//48, 1000)
    """signal = (1 - 0.5 * t + 0.1 * t**2 + 0.01 * t**3) + \
             np.sin(2 * np.pi * t) + 0.5 * np.cos(4 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t) + \
             0.5 * np.random.normal(size=len(t))"""  # Ajout de bruit

    for i in range (signals.shape[0]):
        # Ajustement avec un polynôme de degré 5 et 5 composantes périodiques
        params, fitted_signal = fit_signal_with_trend_and_periodic(t, signals[i,:], poly_degree=5, num_periodic=4)
        print("Paramètres ajustés :", params)


# Ajustement du signal

#params, fitted_signal = fit_signal_with_trend_and_periodic(np.arange(1000), signals[0,:], num_periodic=1, plot_result=True)



# %% Calculating the FFT of the signal
def calculate_fft(signal, sampling_rate):
    """
    Calcule la Transformée de Fourier Rapide (FFT) d'un signal.

    Paramètres :
    - signal : ndarray, le signal d'entrée.
    - sampling_rate : float, la fréquence d'échantillonnage en Hz.

    Retour :
    - frequencies : ndarray, les fréquences associées.
    - amplitude : ndarray, l'amplitude spectrale.
    """
    # Nombre de points dans le signal
    N = len(signal)

    # Calcul de la FFT
    fft_values = np.fft.fft(signal)

    # Calcul des fréquences associées
    frequencies = np.fft.fftfreq(N, d=1 / sampling_rate)

    # On ne conserve que la moitié des fréquences (partie positive du spectre)
    positive_freq_indices = frequencies > 0
    frequencies = frequencies[positive_freq_indices]
    amplitude = np.abs(fft_values[positive_freq_indices]) / N

    return frequencies, amplitude

# Exemple d'utilisation
if __name__ == "__main__":
    # Paramètres du signal
    sampling_rate = 100  # Fréquence d'échantillonnage (Hz)
    duration = 10  # Durée du signal (secondes)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)  # Temps

    for i in range (signals.shape[0]):
        # Calcul de la FFT
        frequencies, amplitude = calculate_fft(signals[i,:], sampling_rate)

        # Affichage du signal original
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, signals[i,:], label="Signal original")
        plt.xlabel("Temps (s)")
        plt.ylabel("Amplitude")
        plt.title("Signal dans le domaine temporel")
        plt.legend()

        # Affichage du spectre de fréquence
        plt.subplot(2, 1, 2)
        plt.plot(frequencies, amplitude, label="Amplitude spectrale", color="red")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Amplitude")
        plt.title("Spectre de fréquence (FFT)")
        plt.legend()
        plt.tight_layout()
        plt.show()

# %% Test 2 to fit temporal series
from scipy.optimize import curve_fit
import scipy.signal as signal
import matplotlib.pyplot as plt

def lowpass_filter(data, cutoff_freq, sample_rate, order=4, plot_result=True):
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

    # Normalisation de la fréquence de coupure (entre 0 et 1)
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist

    # Conception du filtre passe-bas (Butterworth)
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)

    # Application du filtre au signal
    filtered_data = signal.filtfilt(b, a, data)

    # Optionnel : tracer les résultats
    if plot_result:
        plt.figure(figsize=(10, 6))
        plt.plot(data, label="Signal original", alpha=0.7)
        plt.plot(filtered_data, label=f"Signal filtré (fc={cutoff_freq} Hz)", color='red')
        plt.legend()
        plt.xlabel("Temps (échantillons)")
        plt.ylabel("Amplitude")
        plt.title("Filtrage passe-bas")
        plt.show()

    return filtered_data
def fit_signal_with_trend_and_periodic(t, signal, cutoff_freq = 1 , poly_degree=3, num_periodic=2, plot_result=True):
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

    # Définition des paramètres initiaux
    p0 = [0] * (poly_degree + 1)
    
    
    for i in range(num_periodic):
        p0 += [1, cutoff_freq/num_periodic*(i+1), 0]  # Amplitude initiale, fréquence initiale, phase initiale

    # Ajustement des paramètres avec curve_fit
    params, _ = curve_fit(model, t, signal, p0=p0, maxfev=10000)

    # Signal ajusté
    fitted_signal = model(t, *params)

    # Optionnel : afficher les résultats
    if plot_result:
        plt.figure(figsize=(10, 6))
        plt.plot(t, signal, label="Signal original", alpha=0.7)
        plt.plot(t, fitted_signal, label="Signal ajusté (modèle)", linestyle="--", color="red")
        plt.legend()
        plt.xlabel("Temps")
        plt.ylabel("Amplitude")
        plt.title(f"Ajustement : Tendance polynomiale (degré {poly_degree}) + {num_periodic} composantes périodiques (cosinus)")
        plt.show()

    return params, fitted_signal

# Exemple d'utilisation
if __name__ == "__main__":
    # Génération d'un signal synthétique
    sampling_rate = 48
    t = np.linspace(0, 1000//sampling_rate, 1000)
    fc = 2  # Fréquence de coupure du filtre passe-bas
    # Ajustement avec un polynôme de degré 3 et 2 composantes périodiques
    for i in range(signals.shape[0]) :
        filtered_signal = lowpass_filter(signals[i,:], cutoff_freq=fc, sample_rate=sampling_rate, plot_result=False)
        params, fitted_signal = fit_signal_with_trend_and_periodic(t, filtered_signal,cutoff_freq = fc, poly_degree=4, num_periodic=4)
    print("Paramètres ajustés :", params) 
    
    #
    
    
    
   
    

# %% Use GAM 1 to fit temporal series
from pygam import LinearGAM, s


def fit_gam(t, signal, n_splines=20, lam=10, periodic_frequencies=None, plot_result=True):
    """
    Ajuste un modèle GAM sur un signal donné, avec une tendance (splines)
    et des termes périodiques optionnels.

    Paramètres :
    - t : ndarray, les points temporels.
    - signal : ndarray, le signal à ajuster.
    - n_splines : int, nombre de splines pour la tendance.
    - lam : float, paramètre de régularisation pour les splines.
    - periodic_frequencies : list or None, fréquences des termes périodiques à inclure (en Hz).
    - plot_result : bool, affiche le graphique des résultats si True.

    Retour :
    - gam : Modèle GAM ajusté.
    - fitted_signal : ndarray, signal ajusté par le modèle.
    """

    # Construction des caractéristiques
    X = t.reshape(-1, 1)  # Colonne pour la tendance
    if periodic_frequencies is not None:
        for freq in periodic_frequencies:
            X = np.column_stack((X, np.sin(2 * np.pi * freq * t), np.cos(2 * np.pi * freq * t)))

    # Création du modèle GAM
    num_features = X.shape[1]
    if periodic_frequencies is not None:
        # Modèle avec splines pour la tendance + termes linéaires pour périodicité
        gam = LinearGAM(s(0, n_splines=n_splines, lam=lam) + 
                        sum([s(i, n_splines=5) for i in range(1, num_features)])).fit(X, signal)
    else:
        # Modèle uniquement avec splines pour la tendance
        gam = LinearGAM(s(0, n_splines=n_splines, lam=lam)).fit(X, signal)

    # Prédiction du signal ajusté
    fitted_signal = gam.predict(X)

    # Optionnel : afficher les résultats
    if plot_result:
        plt.figure(figsize=(12, 6))
        plt.plot(t, signal, label="Signal original", alpha=0.7)
        plt.plot(t, fitted_signal, label="Signal ajusté (GAM)", linestyle="--", color="red")
        plt.xlabel("Temps")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title(f"Ajustement GAM : {n_splines} splines + {len(periodic_frequencies) if periodic_frequencies else 0} termes périodiques")
        plt.show()

    return gam, fitted_signal


# Exemple d'utilisation
"""if __name__ == "__main__":
    # Génération d'un signal synthétique
    t = np.linspace(0, 1000//48, 1000)
    for i in range(signals.shape[0]) :
        # Ajustement avec 20 splines et 3 composantes périodiques
        gam, fitted_signal = fit_gam(t, signals[i,:], n_splines=20, lam=10, periodic_frequencies=[1, 2, 4])
    print(gam.summary())"""



# %% test de graph_series_analyzer
%load_ext autoreload
%autoreload 2
from graph_series_analyzer import GraphSeriesAnalyzer




# Create a random adjacency matrix
def keep_top_n(W, N: int) -> np.ndarray:
    """ 
    Keep the N largest values in a matrix and set the others to zero.
    
    Args:
        matrix (np.ndarray): Input matrix.
        N (int): Number of largest values to keep.
    returns:
        np.ndarray: Filtered matrix with only the N largest values.
    """
    matrix = W
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
    return filtered_matrix

for i in range (10) :
    W = np.random.rand(12, 12)
    np.fill_diagonal(W, 0)
    W_filtred = keep_top_n(W, 20)
    analyzer = GraphSeriesAnalyzer (W_filtred, data.nodes_dataframe, "temp", 1000 , cutoff_freq= 1 , **df_pos)
    print(analyzer.smoothness)
    print(analyzer.params)
    #print(analyzer.Lsmoothness)

for key in dictW.keys():
    analyzer = GraphSeriesAnalyzer (dictW[key], data.nodes_dataframe, "temp", 1000 , cutoff_freq= 1 , **df_pos)
    print(analyzer.smoothness)
    print(analyzer.params)
    #print(analyzer.Lsmoothness)


# %%
