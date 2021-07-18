# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python [conda env:.conda-mlops]
#     language: python
#     name: conda-env-.conda-mlops-py
# ---

# %% [markdown]
# # Sentiment API

# %%
import pandas as pd
import numpy as np
import sys

# %% [markdown]
# ## Daten einlesen
# Vorbereitete Daten werden eingelesen

# %%
test_df = pd.read_pickle("../data/interim/test_df.pkl")

# %%
test_df.head()

# %% [markdown]
# ## Client Laden
# Der Client wird mittles [`openapi-python-client` Generator](https://github.com/openapi-generators/openapi-python-client) z.B. wie folgt erzeugt.
# ```
# openapi-python-client generate --url http://127.0.0.1:8080/openapi.json
# ```

# %%
sys.path.append("../titanic-survival-model-api-client")
from titanic_survival_model_api_client import Client
from titanic_survival_model_api_client.models import Input, Prediction
from titanic_survival_model_api_client.api.default import predict_predict_post

# %% [markdown]
# Client definieren

# %%
client = Client(base_url="http://127.0.0.1:8080", timeout=30)

# %% [markdown]
# API mit Daten aufrufen und Ergebnisse in DataFrame speichern

# %%
# Loop über zufällige Zeilen des DataFrames
for idx, row in test_df.drop(columns="label").sample(100).iterrows():
    
    # Input Daten erzeugen 
    input = Input.from_dict(row)

    # API aufrufen
    prediction = predict_predict_post.sync(client=client, json_body=input)

    # Daten in DataFrame schreiben
    test_df.loc[idx, "survival"] = prediction.label

# %% [markdown]
# Ausgabe der Sentiments

# %%
test_df[pd.notna(test_df["survival"])]

# %% [markdown]
# ## Outlier
#
# Nun die API testweise mit dem Outlier Datensatz (Passagiere ab einem Alter von 50 Jahren) aufrufen.

# %%
outlier_df = pd.read_pickle("../data/interim/outlier_df.pkl")
# Loop über zufällige Zeilen des DataFrames
for idx, row in outlier_df[(outlier_df["fare"].notna()) & (outlier_df["embarked"].notna())].drop(columns="label").sample(100).iterrows():
    
    # Input Daten erzeugen 
    input = Input.from_dict(row)

    # API aufrufen
    prediction = predict_predict_post.sync(client=client, json_body=input)
