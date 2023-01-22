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
# # Model API

# %%
import pandas as pd
import numpy as np
import sys

# %% [markdown]
# ## Read data
# Prepared data is read.

# %%
test_df = pd.read_pickle("../data/interim/test_df.pkl")

# %%
test_df.head()

# %% [markdown]
# ## Load client
# The client is generated using [`openapi-python-client` generator](https://github.com/openapi-generators/openapi-python-client) e.g. as follows.
# ```
# openapi-python-client generate --url http://127.0.0.1:8080/openapi.json
# ```

# %%
sys.path.append("../titanic-survival-model-api-client")
from titanic_survival_model_api_client import Client
from titanic_survival_model_api_client.models import Input, Prediction
from titanic_survival_model_api_client.api.default import predict_predict_post

# %% [markdown]
# Instantiate client

# %%
client = Client(base_url="http://127.0.0.1:8080", timeout=30)

# %% [markdown]
# Call API with data and save results in DataFrame

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
# Model Output

# %%
test_df[pd.notna(test_df["survival"])]

# %% [markdown]
# ## Outlier
#
# Now call the API on a test basis with the Outlier dataset (passengers aged 50 and over).

# %%
outlier_df = pd.read_pickle("../data/interim/outlier_df.pkl")
# Loop über zufällige Zeilen des DataFrames
for idx, row in outlier_df[(outlier_df["fare"].notna()) & (outlier_df["embarked"].notna())].drop(columns="label").sample(100).iterrows():
    
    # Input Daten erzeugen 
    input = Input.from_dict(row)

    # API aufrufen
    prediction = predict_predict_post.sync(client=client, json_body=input)

# %%
