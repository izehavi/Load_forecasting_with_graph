import matplotlib.pyplot as plt
positions_noeuds = [
    [5.5, 4.5],
    [4.8, 5.0],
    [2.5, 4.0],
    [4.5, 6.0],
    [7.0, 6.5],
    [6.5, 8.0],
    [5.0, 7.0],
    [5.0, 8.5],
    [3.0, 2.5],
    [4.0, 2.0],
    [3.0, 3.5],
    [5.5, 2.0]
]
class plot_graph:
    def __init__(self, G, title="Graphe"):
        self.G = G
        self.title = title
        self.display_graph()
        
    def display_graph(self):
        """Affiche un graphe simple."""
        self.G.plot()
        plt.title(self.title)
        plt.show()

    def display_graph_with_signal(self, signal, title="Graphe avec signal"):
        """Affiche un graphe avec un signal coloré sur les nœuds."""
        self.G.plot_signal(signal, vertex_size=50)
        plt.title(title)
        plt.show()

