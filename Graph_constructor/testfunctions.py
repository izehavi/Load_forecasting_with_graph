""" In this document we will test the functions of the folder (dataloader.py, graph_builder.py)"""
 #%%
import pandas as pd
import numpy as np
from dataloader import DataLoader
from graph_builder import GraphBuilder
from visualizations import GraphPlotter
from graph_analyzer import GraphAnalyzer

#%% config informations 
path_train = r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\Graph_constructor\train.csv"
path_test = r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\Projet\Graph_constructor\test.csv"

L_path_to_W = [
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\examples\graph_representations\correlation\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\examples\graph_representations\distsplines\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\examples\graph_representations\distsplines2\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\examples\graph_representations\dtw\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\examples\graph_representations\gl3sr\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\examples\graph_representations\precision\W.txt",
    r"C:\Users\zehav\OneDrive\Bureau\ENS\S5_ARIA\stage_3mois_graph\Projet_github\examples\graph_representations\space\W.txt"
]


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

#%% Test the DataLoader class
data = DataLoader(path_train, path_test, kwargs={"start_date": "2018-01-01", "end_date": None})


print("hello")
print(data.nodes_dataframe["0"]["temp"][3])
for key in data.nodes_dataframe.keys():
    print(key)

#%% Test the GraphBuilder class
Graph = GraphBuilder(L_path_to_W)

for i in range(len(L_path_to_W)):
    W = Graph.load_Wmatrix(L_path_to_W[i])
    print(W)





    W_threshold = Graph.keep_top_n(W, 60)
    # Test the GraphPlotter class
    plotter = GraphPlotter(W_threshold, df_pos)
    #plotter.plot_graph(title="Graph", figsize=(10, 8))
    #plotter.plot_graph_on_map(title="Graph on Map")




#%% Test the GraphAnalyzer class


for i in range(10) :
    analyzed_graph = GraphAnalyzer(W_threshold, data.nodes_dataframe, "temp" ,i, **df_pos)
    print(analyzed_graph.smoothness)

print("END")


# %%
print("hello")