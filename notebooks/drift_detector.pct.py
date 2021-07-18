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
# # Drift Detector
# https://github.com/SeldonIO/alibi-detect/blob/master/examples/cd_chi2ks_adult.ipynb
# ```
# dvc run -n drift_model -d ../data/interim/train_df.pkl -d ../data/interim/test_df.pkl -d ../data/interim/outlier_df.pkl -d ../models/model.pkl -d ../models/feat_names.json -o ../models/drift_detector.pkl -w notebooks python drift_detector.pct.py
# ```
#

# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import joblib
from scipy.stats import chi2_contingency, ks_2samp

from sklearn.pipeline import Pipeline
from joblib import load

# %% [markdown]
# Daten laden

# %%
train_df = pd.read_pickle("../data/interim/train_df.pkl")
test_df = pd.read_pickle("../data/interim/test_df.pkl")
X_train = train_df.drop(columns=["label"])
X_valid = test_df.drop(columns=["label"])
X_train.tail()

# %% [markdown]
# Pipeline laden

# %%
clf = load("../models/model.pkl")
clf

# %%
preprocessor = Pipeline(clf.steps[:-1])

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
class DriftDetector:
    def __init__(self, X_train, preprocessor, feat_names, categorical_features):
        self.preprocessor = preprocessor
        self.X_train = self._preprocess(X_train)
        self.feat_names = feat_names
        self.categorical_features = categorical_features
    
    def _preprocess(self, X):
        return self.preprocessor.transform(X)
    
    def test(self, X):
        X = self._preprocess(X)
        p_values = []
        for feat in range(self.X_train.shape[1]):
            if feat not in self.categorical_features:
                p_values.append(ks_2samp(self.X_train[:, feat], X[:, feat]).pvalue)
            else:
                p_values.append( 
                      chi2_contingency(np.vstack([np.bincount(self.X_train[:, feat].astype(int)), 
                                                 np.bincount(X[:, feat].astype(int))]))[1])
        return p_values


# %%
dd = DriftDetector(X_train, preprocessor, feat_names, categorical_features)

# %%
dd.test(X_valid)

# %%
outlier_df = pd.read_pickle("../data/interim/outlier_df.pkl")
list(zip(feat_names, dd.test(outlier_df[~outlier_df["fare"].isna()].drop(columns=["label"]))))

# %%
joblib.dump(dd, "../models/drift_detector.pkl")
