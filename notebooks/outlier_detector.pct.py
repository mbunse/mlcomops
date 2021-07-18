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
# # Outlier Detector
# ```
# dvc run -n outlier_detector --force -d ../data/interim/train_df.pkl -d ../data/interim/test_df.pkl -d ../models/feat_names.json -d ../models/model.pkl ../data/interim/outlier_df.pkl -o ../models/outlier_detector.pkl -w notebooks python outlier_detector.pct.py
# ```

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from joblib import load

# %% [markdown]
# Daten laden

# %%
train_df = pd.read_pickle("../data/interim/train_df.pkl")
test_df = pd.read_pickle("../data/interim/test_df.pkl")

# %% [markdown]
# Pipeline laden

# %%
pipeline = load("../models/model.pkl")
preprocessor = Pipeline(pipeline.steps[:-1])

# %%
feat_names = json.load(open("../models/feat_names.json", "r"))
dict(enumerate(feat_names))

# %%
categorical_features=[4, 5, 6]
categorical_names={
    4: ['C', 'Missing', 'Q', 'S'],
    5: ['female', 'male'],
    6: ["1st class", "2nd class", "3rd class"],
}

# %%
od = Pipeline(preprocessor.steps + [("sd", StandardScaler()), ("od", IsolationForest(random_state=12345, contamination=0.02))])
od.fit(train_df.drop(columns=["label"]))

# %%
od = Pipeline(preprocessor.steps + [("sd", StandardScaler()), ("od", OneClassSVM())])
od.fit(train_df.drop(columns=["label"]))

# %%
od.predict(train_df.drop(columns=["label"]).sample(10))

# %%
od.decision_function(train_df.drop(columns=["label"]).sample(10))

# %%
fig, ax = plt.subplots()
pred = od.decision_function(train_df.drop(columns=["label"]))
ax.scatter(x=range(pred.shape[0]), y=pred)
ax.axhline(y=0,c="C1")

# %%
outlier_df = pd.read_pickle("../data/interim/outlier_df.pkl")
fig, ax = plt.subplots()
pred = od.decision_function(outlier_df[~outlier_df["fare"].isna()].drop(columns=["label"]))
ax.scatter(x=range(pred.shape[0]), y=pred)
ax.axhline(y=0,c="C1")

# %%
outlier_df = pd.read_pickle("../data/interim/outlier_df.pkl")
outlier_df = outlier_df[~outlier_df["fare"].isna()]
outlier_df["label"]=1 # is outlier
test_df = pd.read_pickle("../data/interim/test_df.pkl")
test_df["label"]=0 # is inliner
comb_df = pd.concat([outlier_df, test_df])
fig, ax = plt.subplots()
pred = od.decision_function(comb_df.drop(columns=["label"]))
ax.scatter(x=range(pred.shape[0]), y=pred, c=comb_df["label"].map(lambda x: f"C{x}"))
ax.axhline(y=0,c="C1")

# %%
joblib.dump(od, "../models/outlier_detector.pkl")
