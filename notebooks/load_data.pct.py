# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:mlops]
#     language: python
#     name: conda-env-mlops-py
# ---

# %% [markdown]
# # Trainingsdaten aufbereiten
#
# In diesem Notebook werden die Roh-Daten für das Training aufbereitet.
#
# DVC Pipeline wurde mit folgendem Befehl eingerichtet:
# ```
# dvc run -n load_data --force -o ../data/interim/train_df.pkl -o ../data/interim/test_df.pkl -o ../data/interim/outlier_df.pkl -d load_data.pct.py -w notebooks python load_data.pct.py
# ```

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# %% [markdown]
# ## Daten einlesen
#
# Der [Titanic Datensatz von OpenML](https://www.openml.org/d/40945) wird eingelesen.

# %%
URL = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
data_df = pd.read_csv(URL, na_values="?")
data_df.head()

# %% [markdown]
# Features
#
# | Spalte | Inhalt |
# | --- | --- |
# | survival | Überleben (0 = Nein; 1 = Ja)
# | class | Passagier Klasse (1 = 1. Klasse; 2 = 2. Klasse; 3 = 3. Klasse)
# | name | Name
# | sex | Geschlecht
# | age | Alter
# | sibsp | Anzahl Geschwister/Ehepartner an Bord der Titanic
# | parch | Anzahl Eltern/Kinden an Bord der Titanic
# | ticket | Ticket Nummer
# | fare | Fahrpreis
# | cabin | Kabine
# | embarked | Einschiffungshafen (C = Cherbourg (FR); Q = Queenstown (IR); S = Southampton (EN))
# | boat | Rettungsboot (falls überlebt)
# | body | Leichnam Nummer (wenn nicht überlebt und Leichnam geborgen wurde)

# %% [markdown]
# Die Spalte `survival` wird als Label gesetzt. Die Spalten `boat` und `body` werden entfernt, da sie direkt mit dem Label verknüpft sind.

# %%
model_df = data_df.rename(columns={"survived": "label"})
model_df = model_df.drop(columns=["boat", "body"])
model_df.head()

# %% [markdown]
# Für das Training sollen zunächst nur jüngere Passagiere genutzt werden. Die Verteilung sieht wie folgt aus:

# %%
ax = model_df["age"].plot.hist(bins=np.arange(0,100,10))
ax.set_xlabel("Alter");

# %%
outlier_df = model_df[model_df["age"]>=50]
model_df = model_df[model_df["age"]<50]
model_df.head()

# %% [markdown]
# ## Daten aufbereiten
#

# %%
train_df, test_df = train_test_split(model_df, random_state=12345, test_size=0.2)
len(train_df), len(test_df)

# %% [markdown]
# ## Aubereitete Daten ausgeben
#
# Der DataFrame wird dann als Pickle ausgeben.
#

# %%
os.makedirs("../data/interim", exist_ok=True)

# %%
train_df.to_pickle("../data/interim/train_df.pkl")
test_df.to_pickle("../data/interim/test_df.pkl")
outlier_df.to_pickle("../data/interim/outlier_df.pkl")
