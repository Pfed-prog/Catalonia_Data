#!/usr/bin/env python
# coding: utf-8

# In[11]:


"""script"""
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


def get_input(local=False):
    if local:
        print("Reading local file 9c820e0e5b3a4264aa5058f24a82386d.csv")

        return "9c820e0e5b3a4264aa5058f24a82386d.csv"

    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")

        return filename


def run_linear_regression(local=False):
    filename = get_input(local)
    if not filename:
        print("Could not retrieve filename.")
        return

    df = pd.read_csv(filename, nrows=10000).dropna()
    
    cols = ['01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
       '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
       '19h', '20h', '21h', '22h', '23h', '24h']
    
    df.DATA = pd.to_datetime(df.DATA, format="%d/%m/%Y")

    df['year'] = df.DATA.dt.year
    df['month'] = df.DATA.dt.month
    df['day'] = df.DATA.dt.day

    data = df[df['MAGNITUD'] == 8].reset_index()

    le = preprocessing.LabelEncoder()
    le.fit(data['NOM ESTACIO'])

    data['NOM ESTACIO'] = le.transform(data['NOM ESTACIO'])

    regr = RandomForestRegressor(n_estimators=300,  random_state=0)

    regr.fit(data[['month', 'year', 'NOM ESTACIO'] + cols ], data[cols])

    Estacio_cols = ['Badalona', 'Barcelona (Poblenou)', 'Barcelona (St. Gervasi)',
       "L'Hospitalet de Llobregat", 'Montcada i Reixac',
       'Sant Adrià de Besòs', 'Vallcebre', 'Cercs (St. Corneli)',
       'la Nou de Berguedà (Malanyeu)', 'Constantí', 'Manresa',
       'Perafort (Puigdelfí)', 'Tarragona (Bonavista)',
       'Tarragona (pl. Generalitat)', 'Vila-seca',
       'la Pobla de M./el Morell', 'Tarragona (Sant Salvador)',
       'Igualada', 'Martorell', 'Terrassa', 'Vic', 'Sarrià de Ter',
       'Granollers (av. Joan Prim)', 'Mollet del Vallès', 'Reus',
       'Mataró', 'Barcelona (Sagrera)', 'Cercs (St. Jordi)', 'Lleida',
       'Sabadell (pl. Creu de Barberà)', 'Sant Fost de Campsentelles',
       'Sabadell', 'Sant Celoni', 'Rubí', 'Sta. Coloma de Gr. (c/ Bruc)',
       'Sant Cugat del Vallès', 'Tarragona (Universitat Laboral)',
       'Vilanova i la Geltrú', 'Fornells de la Selva (escola municipal)',
       'Barcelona (Sants)', 'Granollers (c/ Joan Vinyoli)',
       'Sta. Perpètua de Mogoda', 'Vilafranca del Penedès',
       'Barcelona (Eixample)', 'Santa Coloma de Gramenet',
       'Barcelona (Gràcia - Sant Gervasi)', 'Barberà del Vallès',
       'Sant Andreu de la Barca', 'el Prat de Llobregat (església)',
       'Sant Vicenç dels Horts (Ribot)', 'Gavà (c/Girona - c/Progrés)',
       'Cornellà de Llobregat (Allende - Bonveí)',
       'Tarragona (Parc de la Ciutat)', 'Cercs (Sant Jordi)',
       'Bellver de Cerdanya', 'Barcelona (Ciutadella)',
       'Girona (parc de la Devesa)', 'Gavà', 'Cubelles (Poliesportiu)',
       'Tona', 'Alcover', 'Vallcebre (campanar)',
       'Santa Perpètua de Mogoda', 'Castellet i la Gornal',
       'Cercs (Sant Corneli)', 'Vandellòs (Els Dedalts)',
       'Vandellòs (Viver)', 'Berga', 'Barcelona (Parc Vall Hebron)',
       'Montseny (La Castanya)', 'Granollers', 'Viladecans - Atrium',
       'el Prat de Llobregat (Sant Cosme)', 'Tona (Zona Esportiva)',
       "L'Ametlla de Mar", 'Sta. Margarida i els Monjos (La Ràpita)',
       'El Prat de Llobregat (Jardins de la Pau)', 'Amposta',
       'Sitges (Vallcarca)', 'Vandellòs (Barranc del Terme)',
       'Barcelona (Torre Girona)', 'Manlleu', 'Montsec',
       'El Prat de Llobregat (Sagnier)', 'Barcelona (Palau Reial)',
       'Girona (Escola de Música)', 'Pallejà (Roca de Vilana)', 'Alcanar',
       'Sant Vicenç dels Horts', 'Sant Feliu de Ll. (CEIP Marti i Pol)',
       'Sitges (Vallcarca - Oficines)', 'Juneda (Pla del Molí)', 'Begur',
       'Santa Pau', 'Barcelona (Observatori Fabra)',
       'Vila-seca (IES Vila-seca)']
    
    df_predict = pd.DataFrame({"day": [x for x in range(15, 29)] * 96, "NOM ESTACIO": Estacio_cols * 14 })
    
    df_predict['NOM ESTACIO'] = le.transform(df_predict['NOM ESTACIO'])
    
    predictions = regr.predict(df_predict[['month', 'year', 'NOM ESTACIO']])
    
    print(predictions)
    filename = "logistic_regression.pickle" if local else "/data/outputs/result"

    with open(filename, "wb") as pickle_file:
        print(f"Pickling results in {filename}")
        pickle.dump(predictions, pickle_file)


if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_linear_regression(True)


# In[ ]:




