import numpy as np
from pygsp import filters

def add_signal_with_noise(G, signal_strength=1, noise_level=0.5):
    """Ajoute un signal aléatoire avec du bruit."""
    rs = np.random.RandomState(42)
    signal = np.zeros(G.N)
    signal[: G.N // 2] = signal_strength
    noise = rs.uniform(-noise_level, noise_level, size=G.N)
    return signal + noise

def apply_low_pass_filter(G, signal, tau=10):
    """Applique un filtre passe-bas à un signal sur un graphe."""
    def low_pass(x):
        return 1 / (1 + tau * x)

    g_filter = filters.Filter(G, low_pass)
    filtered_signal = g_filter.filter(signal)
    return filtered_signal

def smoothness_calcul(G, signal):
    return signal.T @ G.L @ signal
