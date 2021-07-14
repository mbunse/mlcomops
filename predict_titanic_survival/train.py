#!/bin/python
"""Skript um Training des Modells durchzuführen. 
Das Model wird nach `model/model.pkl` persistiert.

dvc Pipeline anlegen:
```
dvc run -d data\\interim\\model_dev_data.pkl -d nuedigex\\train.py 
    -M score.json -o models/model.pkl python -m nuedigex.train
```
Ausführen mit
```
python -m neudigex.train
```
"""
import logging
import datetime
import json

import pandas as pd 
import joblib
import mlflow
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV, train_test_split

from hamlops import model, report

LOG = logging.getLogger(__name__)

def train():
    """Funktion um Modell zu traininieren"""

    # set tracking url to central file
    mlflow.set_tracking_uri("mlruns/")
    mlflow.sklearn.autolog()

    mlflow.set_experiment('New experiment')

    # Einlesen der Daten
    filepath = "data/interim/model_dev_data.pkl"
    
    data_df = pd.read_pickle(filepath)
    labels = data_df["label"].copy()
    features = data_df.drop(columns=["label"]).copy()

    del data_df

    # Aufteilen in Training (75%) und Test (25%)
    features_train, features_test,  \
        labels_train, labels_test = \
        train_test_split(features, labels, 
            	         test_size=0.1, train_size=0.1, 
                         random_state=42, stratify=labels)

    # Gewichtung bestimmen
    sample_weight = compute_sample_weight("balanced", labels_train)

    # Modell-Pipeline wie in model.py definiert
    clf = model.PIPELINE_FULL

    # Beispiel Parameter-Grid
    param_grid = {
        "Classifier__max_depth": [2, 3],
        "Classifier__min_samples_leaf": [5, 20]
    }
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=4, n_jobs=4)

    start = datetime.datetime.now()
    logging.info("Starting fitting")
    
    with mlflow.start_run() as run:
        # Grid-Search unter Berücksichtigung der Sample-Weights durchführen
        grid_search.fit(features_train, labels_train, 
            **{"Classifier__sample_weight": sample_weight})

    end = datetime.datetime.now()
    logging.info("Fitting took %s", end - start)

    # Ausgabe eines Reports für Grid-Search
    score = report.report(grid_search, features_train, labels_train, features_test, labels_test)
    with open("score.json", "w") as f:
        json.dump(score, f)

    # Auf allen Daten Trainieren
    sample_weight = compute_sample_weight("balanced", labels)
    clf.set_params(**grid_search.best_params_)
    clf.fit(features, labels, **{"Classifier__sample_weight": sample_weight})

    # Modell speichern
    joblib.dump(clf, "models/model.pkl")

if __name__ == "__main__":
    train()
