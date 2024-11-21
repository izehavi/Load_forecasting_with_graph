from generate_graph import generate_random_graph, generate_ring_graph, construct_real_graph
from display_graph import plot_graph
from process_graph import add_signal_with_noise, apply_low_pass_filter, smoothness_calcul
from real_graph_generator import RealGraphGenerator

L_path_to_W = ["examples/graph_representations/correlation/W.txt", "examples/graph_representations/distsplines/W.txt", 
               "examples/graph_representations/distsplines2/W.txt", "examples/graph_representations/dtw/W.txt",
               "graph_representations/gl3sr/W.txt", "examples/graph_representations/precision/W.txt",
               "examples/graph_representations/space/W.txt"]

def main():
    for path_to_W in L_path_to_W:
        # Générer un graphe aléatoire
        Graph = RealGraphGenerator(path_to_W)
        G = Graph.G
        # Afficher le graphe
        plot_graph(G, title="Graphe réel")

    """# Ajouter un signal avec du bruit
    signal = add_signal_with_noise(G, signal_strength=1, noise_level=0.5)
    display_graph_with_signal(G, signal, title="Signal bruité")

    # Appliquer un filtre passe-bas
    filtered_signal = apply_low_pass_filter(G, signal)
    display_graph_with_signal(G, filtered_signal, title="Signal filtré")
    
    # Calcul de la smoothness
    smoothness_signal = smoothness_calcul(G, signal)
    smoothness_filtered_signal = smoothness_calcul(G, filtered_signal)
    print(f"Smoothness du signal bruité : {smoothness_signal}")
    print(f"Smoothness du signal filtré : {smoothness_filtered_signal}")"""

if __name__ == "__main__":
    main()
