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
from joblib import dump

#importer le fichier csv
myData = pd.read_excel("D:\Dataset\Dataodc.xlsx", parse_dates=True)
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
       'Observations', 'date de déclenchement', 'Date de premier rapportage en tant qu\'ODF'], axis=1)


#Data pre-processing
myData = myData.dropna(axis=0)

encoder = LabelBinarizer()
myData['District'] = encoder.fit_transform(myData['District'])
#myData['Village ODF ou non ODF?'] = encoder.fit_transform(myData['Village ODF ou non ODF?'])

#affiche le dataset
print(myData.info())
print(myData.iloc[2:3, :3].to_string())

x = myData.iloc[:,:10].copy()
y = myData.iloc[:,10].copy()
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1) # Split data for test and training

# Paramètres à rechercher pour chaque modèle
svc_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf']
}
rf_param_grid = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
knn_param_grid = {
    'n_neighbors': np.arange(3,20,2),
    'weights': ['uniform', 'distance']
}

print('initialisation des modèles...')
# Initialisation des modèles
svc_model = SVC()
rf_model = RandomForestClassifier()
knn_model = KNeighborsClassifier()

print('succes !')
print('initialisation de GridSearchCV...')
# Initialisation des objets GridSearchCV pour chaque modèle avec leurs paramètres respectifs
#svc_grid = GridSearchCV(estimator=svc_model, param_grid=svc_param_grid, cv=5)
rf_grid = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5)
knn_grid = GridSearchCV(estimator=knn_model, param_grid=knn_param_grid, cv=5)

print('succes !')
print('entrainnement...')
# Adapter les grilles de recherche aux données
#svc_grid.fit(x_train, y_train)
rf_grid.fit(x_train, y_train)
knn_grid.fit(x_train, y_train)

# Afficher les meilleurs paramètres et scores pour chaque modèle
#print("SVC - Meilleurs paramètres:", svc_grid.best_params_)
#print("SVC - Meilleur score:", svc_grid.best_score_)
print("\nRandom Forest - Meilleurs paramètres:", rf_grid.best_params_)
print("Random Forest - Meilleur score:", rf_grid.best_score_)
print("\nKNeighbors - Meilleurs paramètres:", knn_grid.best_params_)
print("KNeighbors - Meilleur score:", knn_grid.best_score_)

#model_svc = svc_grid.best_estimator_
model_rf = rf_grid.best_estimator_
model_knn = knn_grid.best_estimator_

print(model_rf.score(x_test, y_test))
print(model_knn.score(x_test, y_test))

#dump(model_svc, 'model_svc.joblib')
dump(model_rf, 'model_rf.joblib')
dump(model_knn, 'model_knn.joblib')

