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
# # Sentiment API

# %%
import pandas as pd
import numpy as np
import sys

# %% [markdown]
# ## Daten einlesen
# Vorbereitete Daten werden eingelesen

# %%
data_df = pd.read_csv("../data/raw/transport-short.csv", header=None, nrows=1000, names=['id', 'kind', 'title', 'link_id', 'parent_id', 'ups', 'downs', 'score',
       'author', 'num_comments', 'created_utc', 'permalink', 'url', 'text',
       'level', 'top_parent'])

# %%
data_df.head()

# %% [markdown]
# ## Client Laden
# Der Client wird mittles [`openapi-python-client` Generator](https://github.com/openapi-generators/openapi-python-client) z.B. wie folgt erzeugt.
# ```
# openapi-python-client generate --url http://127.0.0.1:8080/openapi.json
# ```

# %%
sys.path.append("../sentiment-model-api-client")
from sentiment_model_api_client import Client
from sentiment_model_api_client.models import Input, Sentiment
from sentiment_model_api_client.api.default import predict_post

# %% [markdown]
# Client definieren

# %%
client = Client(base_url="http://127.0.0.1:8080", timeout=30)

# %% [markdown]
# API mit Daten aufrufen udn Ergebnisse in DataFrame speichern

# %%
# Loop über zufällige Zeilen des DataFrames
for idx, row in data_df.sample(500).iterrows():
    
    # Ausschließen von uninteressanten Zeilen
    if not pd.isna(row["text"]) and row["text"] not in ["[deleted]", "[removed]"]:
        
        # Input Daten erzeugen 
        input = Input(sentence=row["text"])
        
        # API aufrufen
        sentiment = predict_post.sync(client=client, json_body=input)
        
        # Daten in DataFrame schreiben
        data_df.loc[idx, "sentiment"] = sentiment.label

# %% [markdown]
# Ausgabe der Sentiments

# %%
data_df[pd.notna(data_df["sentiment"])][["text", "sentiment"]]
