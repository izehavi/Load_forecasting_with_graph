import pandas as pd

model_kwargs = {
    'model_name': 'GraphSAGE',
    'lr': 0.001,
    'num_epochs': 200,
    'hidden_channels': 97,
    'out_channels': 1,
    'num_layers': 1,
    'batch_size': 314,
}

data_kwargs = {
    'day_sup_train': '2018-01-01',
    'day_inf_val': '2018-01-01',
    'node_var': 'Region',
    'dummies': ['Instant', 'JourSemaine', 'DayType', 'offset']
}

dataset_kwargs = {
    'adj_matrix': 'dtw',
    'batch_size': 512,
    'features_base': ['temp', 'nebu', 'wind', 'tempMax', 'tempMin', 'Posan', 'Instant', 'JourSemaine', 'JourFerie', 'offset', 'DayType', 'Weekend', 'temp_liss_fort', 'temp_liss_faible', 'DayValidity'],
    'target_base': 'load',
}

optim_kwargs = {
    'num_layers': (1, 5),
    'hidden_channels': (32, 128),
    'batch_size': (256, 1024),
    'heads': (1, 8)
}

explain_kwargs = {
    'months': {0: 'January', 7200: 'June', 10128: 'August', 14496: 'November'},
}

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
    }
).sort_values(by='REGION').reset_index(drop=True)