#!/bin/python
"""
Skript um Vorhersage des Modells zu testen

Ausführen:
```
python predict.py
```
"""

import joblib
import pandas as pd

def predict():
    """Beispiel Funktion zur Vorhersage mit 
    trainiertem Modell"""
    clf = joblib.load("models/model.pkl")

    # Einlesen der Daten
    filepath = "data/interim/model_dev_data.pkl"

    data_df = pd.read_pickle(filepath)
    labels = data_df["label"].copy()
    features = data_df.drop(columns=["label"]).copy()

    print(features.iloc[0:1])

    print(f"label:      {labels.iloc[0]}")
    print(f"prediction: {clf.predict(features.iloc[0:1])[0]}")

if __name__ == "__main__":
    predict()
