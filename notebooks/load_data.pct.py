# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python [conda env:mlops]
#     language: python
#     name: conda-env-mlops-py
# ---

# %% [markdown]
# # Prepare training data
#
# Run the DVC pipeline with the following command:
# ```
# dvc run -n load_data --force -o ../data/interim/train_df.pkl -o ../data/interim/test_df.pkl -o ../data/interim/outlier_df.pkl -d load_data.pct.py -w notebooks python load_data.pct.py
# ```

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## Load Data
#
# Load [Titanic data set from OpenML](https://www.openml.org/d/40945).

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
# | home.dest | Home/Destination

# %% [markdown]
# The column `survival` is set as label. The columns `boat` and `body` are removed, because they are directly linked to the label.

# %%
model_df = data_df.rename(columns={"survived": "label"})
model_df = model_df.drop(columns=["boat", "body"])
model_df.head()

# %% [markdown]
# Initially, only younger passengers will be used for training. The distribution is as follows:

# %%
ax = model_df["age"].plot.hist(bins=np.arange(0,100,10))
ax.set_xlabel("Alter");

# %% [markdown]
# All individuals 50 and over will be offloaded to an "outlier" dataset and not used for training, validation, and testing.

# %%
outlier_df = model_df[model_df["age"]>=50]
model_df = model_df[model_df["age"]<50]
model_df.head()

# %% [markdown]
# ## Split train and test data set
#
# Withhold a test data set that will not be used for training and will only be used for a final model check at the end.

# %%
train_df, test_df = train_test_split(model_df, random_state=12345, test_size=0.2, stratify=model_df["label"])
len(train_df), len(test_df)

# %% [markdown]
# ## Save data
#
# The DataFrames are output as a pickle.

# %%
os.makedirs("../data/interim", exist_ok=True)

# %%
train_df.to_pickle("../data/interim/train_df.pkl")
test_df.to_pickle("../data/interim/test_df.pkl")
outlier_df.to_pickle("../data/interim/outlier_df.pkl")
