# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#importer le fichier csv
myData = pd.read_excel("D:\ODC\Projet\Dataset\Dataodc.xlsx")
myData = myData.drop(['N°', 'NOM AMO', 'Region', 'Commune', 'Fokontany',
       'Localité (village)', 'Nombre de latrine existante avant intervention',
       'Milieu (rural/urbain)', 'Date d\'auto-déclartion ODF par la communauté',
       '\nDate de dernièr suivi/vérification', 'MOIS DU DERNIER RAPPORTAGE ',
       '# ménage du village', '# population totale',
       'Longitude (dégré décimal)', 'latitude (dégré décimal)', 'ALTITUDE',
       '# latrines en cours de construction',
       '# de ménages ayant/utilisant DLM (dans le foyer ou pres de latrine)',
       '# LN emmergeants', '# personnes touchées par FuM',
       '# personnes touchées par IEC/CCC', '# personnes vulnérables',
       'population totale de la région', 'taux d\'acces par région', 'année',
       'Nom des LN', 'N° tel des LN ou du village', 'Nom du CC responsable',
       'N° tel de CC ',
       'Nom et responsabilité du Champion identifié (A mettre dans le village le plus proche de son domicile habituel)',
       'Contact du Champion', 'Nom de TA responsable', 'N° tel de TA ',
       'Appréciation par rapport à la cohérence des données rapportées',
       'Classification subjective du village ODF (JAUNErouge vert)',
       'Observations'], axis=1)


#Data pre-processing
myData = myData.dropna(axis=0)

encoder = LabelBinarizer()
myData['District'] = encoder.fit_transform(myData['District'])
#myData['Village ODF ou non ODF?'] = encoder.fit_transform(myData['Village ODF ou non ODF?'])

#affiche le dataset
print(myData.info())
print(myData.iloc[2:3, :3].to_string())

x = myData.iloc[:,:12].copy()
y = myData.iloc[:,12].copy()
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1) # Split data for test and training

# Paramètres à rechercher pour chaque modèle
svc_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
knn_param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

# Initialisation des modèles
svc_model = SVC()
rf_model = RandomForestClassifier()
knn_model = KNeighborsClassifier()

# Initialisation des objets GridSearchCV pour chaque modèle avec leurs paramètres respectifs
svc_grid = GridSearchCV(estimator=svc_model, param_grid=svc_param_grid, cv=5)
rf_grid = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5)
knn_grid = GridSearchCV(estimator=knn_model, param_grid=knn_param_grid, cv=5)

# Adapter les grilles de recherche aux données
svc_grid.fit(x_train, y_train)
rf_grid.fit(x_train, y_train)
knn_grid.fit(x_train, y_train)








