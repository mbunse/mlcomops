#!/bin/python
# -*- coding: utf-8 -*-
# %% [markdown]
# # Model Training
# Das Model wird nach `model/model.pkl` persistiert.
#
# dvc Pipeline anlegen:
# ```
# dvc run -n train --force -d ../data/interim/train_df.pkl -d train.pct.py -M ../models/score.json -o ../models/model.pkl -o ../models/feat_names.json -w notebooks python train.pct.py
# ```
#

# %%
import logging
import datetime
import json
import pandas as pd 
import joblib
import mlflow
import os
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

# %%
import sys
sys.path.append("../")
from predict_titanic_survival.data_prep import CustomFeatures
from predict_titanic_survival.report import report

# %% [markdown]
# ## Einlesen der Daten

# %%
filepath = "../data/interim/train_df.pkl"

data_df = pd.read_pickle(filepath)
labels = data_df["label"].copy()
features = data_df.drop(columns=["label"]).copy()

# %%
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, random_state=12345, stratify=labels)
features_train.shape, features_test.shape, labels_train.shape, labels_test.shape

# %% [markdown]
# ## Modell-Pipeline definieren

# %%
clf = Pipeline([
    ("select_feat", ColumnTransformer([
        ("select_num_cols", "passthrough", ["pclass", "sibsp", "parch", "fare", "age"]),
        ("encode_str_cols", Pipeline([
            ("replace_nan", SimpleImputer(strategy="constant", fill_value="Missing")), 
            ("encode", OneHotEncoder(sparse=False, handle_unknown="ignore"))
        ]), ["embarked", "sex"]),
        ("encode_txt_cols", CountVectorizer(binary=True, vocabulary=["Mrs", "Ms", "Mr", "Mme", "Mlle", "Miss"]), "name")
    ], remainder="drop")),
    ("impute", FeatureUnion([
        ("impute", KNNImputer()),
        ("miss_indicator", MissingIndicator()),
    ])),
    ("model", GradientBoostingClassifier(random_state=1234))
])

# %% [markdown]
# ## Feature transformieren

# %%
X_transformed = Pipeline(clf.steps[:-1]).fit_transform(features_train, labels_train)
X_transformed.shape

# %% [markdown]
# Namen der automatisch abgeleiteten Feature extrahieren.

# %%
feat_names_cols = clf.steps[0][1].transformers[0][2] + \
clf.steps[0][1].transformers_[1][1].steps[1][1].get_feature_names().tolist() + \
clf.steps[0][1].transformers_[2][1].get_feature_names()
feat_names = feat_names_cols + [f"miss_ind_{feat_names_cols[i]}" for i in clf.steps[1][1].transformer_list[1][1].features_]
assert len(feat_names)==X_transformed.shape[1]
feat_names

# %%
clf.get_params()

# %% [markdown]
# Funktion um Modell zu traininieren
#
# Tracking der Experimente mittels mlflow

# %%
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.sklearn.autolog()
mlflow.set_experiment('New experiment')

# %%
os.environ["AWS_ACCESS_KEY_ID"]='minio-access-key'
os.environ["AWS_SECRET_ACCESS_KEY"]='minio-secret-key'
os.environ["MLFLOW_S3_ENDPOINT_URL"]='http://localhost:9000'
# export AWS_SECRET_ACCESS_KEY='minio-secret-key'"]

# %%
param_grid = {
    "model__max_depth": [2, 3],
    "model__min_samples_leaf": [5, 20]
}
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=4, n_jobs=4)

with mlflow.start_run() as run:
    sample_weight = compute_sample_weight("balanced", labels_train)
    # Grid-Search unter Berücksichtigung der Sample-Weights durchführen
    grid_search.fit(features_train, labels_train, 
        **{"model__sample_weight": sample_weight})

# %% [markdown]
# Ausgabe eines Reports für Grid-Search

# %%
score = report(grid_search, features_train, labels_train, features_test, labels_test)
score

# %% [markdown]
# Auf allen Daten Trainieren

# %%
sample_weight = compute_sample_weight("balanced", labels)
clf.set_params(**grid_search.best_params_)
clf.fit(features, labels, **{"model__sample_weight": sample_weight})

# %% [markdown]
# ## Ausgaben speichern
#
# Model speichern

# %%
os.makedirs("../models", exist_ok=True)
joblib.dump(clf, "../models/model.pkl")

# %% [markdown]
# Metrik speichern

# %%
with open("../models/score.json", "w") as f:
    json.dump(score, f)

# %% [markdown]
# Feature Namen speichern

# %%
with open("../models/feat_names.json", "w") as f:
    json.dump(feat_names, f)
