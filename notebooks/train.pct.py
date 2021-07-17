#!/bin/python
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
import cloudpickle
import mlflow
import os
import inspect
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics as skm

from fairlearn.reductions import GridSearch
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, true_positive_rate, count

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
pipeline = Pipeline([
    ("select_feat", ColumnTransformer([
        ("select_num_cols", "passthrough", ["sibsp", "parch", "fare", "age"]),
        ("encode_str_cols", Pipeline([
            ("replace_nan", SimpleImputer(strategy="constant", fill_value="Missing")), 
            ("encode", OrdinalEncoder(handle_unknown="error"))
        ]), ["embarked", "sex", "pclass"])
    ], remainder="drop")),
    ("impute", FeatureUnion([
        ("imputed", KNNImputer()),
        ("miss_indicator", MissingIndicator()),
    ])),
    ("clf", GradientBoostingClassifier(random_state=1234))
])

# %% [markdown]
# ## Feature transformieren

# %%
preprocessor = Pipeline(pipeline.steps[:-1])
X_transformed = preprocessor.fit_transform(features_train, labels_train)
X_transformed.shape


# %% [markdown]
# Namen der automatisch abgeleiteten Feature extrahieren.

# %%
def get_feature_names(transformer, parent_feature_names=None):
    feature_names = parent_feature_names
    
    if isinstance(transformer, Pipeline):
        for _, step in transformer.steps:
            feature_names = get_feature_names(step, parent_feature_names=feature_names)
    
    elif isinstance(transformer, FeatureUnion):
        feature_names = []
        for name, trans in transformer.transformer_list:
            if parent_feature_names is None:
                parent_feature_names = name
            feature_names.extend([f"{name}_{item}" for item in get_feature_names(trans, parent_feature_names=parent_feature_names)])
    
  
    elif isinstance(transformer, ColumnTransformer):
        #check_is_fitted(transformer)
        feature_names = []
        for name, trans, column, _ in transformer._iter(fitted=True):
            if trans == "drop" or (hasattr(column, "__len__") and not len(column)):
                continue
            if trans == "passthrough":
                if hasattr(transformer, "_df_columns"):
                    if ((not isinstance(column, slice)) and all(isinstance(col, str) for col in column)):
                        feature_names.extend(column)
                    else:
                        feature_names.extend(transformer._df_columns[column])
                else:
                    indices = np.arange(transformer._n_features)
                    feature_names.extend([f'x{i}' for i in indices[column]])
            else:
                feature_names.extend(get_feature_names(trans, parent_feature_names=column))
    
    elif hasattr(transformer, "get_feature_names"):
        if parent_feature_names is not None and isinstance(parent_feature_names, slice):
            raise ValueError()
        
        if "input_features" in inspect.getfullargspec(transformer.get_feature_names)[0]:
            feature_names = transformer.get_feature_names(input_features=parent_feature_names)
        else:
            feature_names = transformer.get_feature_names()
            if parent_feature_names is not None:
                feature_names = [f'{parent_feature_names}_{feat}' for feat in feature_names]
    
    elif hasattr(transformer, "get_support"):
        if not parent_feature_names:
            raise ValueError()
        mask = transformer.get_support()
        feature_names = (np.array(parent_feature_names)[mask]).tolist()
    elif hasattr(transformer, "features_"): #MissingIndicator
        if not parent_feature_names:
            raise ValueError()
        feature_names = np.array(parent_feature_names)[transformer.features_]
    elif hasattr(transformer, "categories_"):
        feature_names = parent_feature_names
        #         feature_names = [f"{name} " + ",".join([f"{idx}:{cat}" for idx, cat in enumerate(mapping)]) \
        #                          for name, mapping in zip(parent_feature_names, transformer.categories_)]
    elif hasattr(transformer, "clf"):
        feature_names = transformer.clf.classes_.tolist()
    else:
        if parent_feature_names is None:
            raise ValueError(f"{transformer.__class__} does not provice feature names and no parent feature names are given.")
        feature_names = parent_feature_names
    return feature_names

# %%
feat_names = get_feature_names(pipeline)
assert len(feat_names)==X_transformed.shape[1]
feat_names

# %%
pipeline.get_params()

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
    "clf__max_depth": [2, 3],
    "clf__min_samples_leaf": [5, 20]
}
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=4, n_jobs=4, )

with mlflow.start_run() as run:
    sample_weight = compute_sample_weight("balanced", labels_train)
    # Grid-Search unter Berücksichtigung der Sample-Weights durchführen
    grid_search.fit(features_train, labels_train, 
        **{"clf__sample_weight": sample_weight})

# %% [markdown]
# Ausgabe eines Reports für Grid-Search

# %%
score = report(grid_search, features_train, labels_train, features_test, labels_test)
score

# %%
mlflow.sklearn.autolog(disable=True)

# %% [markdown]
# ## Fairness Metriken bestimmen

# %%
y_pred = grid_search.best_estimator_.predict(features_test)

# %%
metrics = {
    'accuracy': skm.accuracy_score,
    'precision': skm.precision_score,
    'recall': skm.recall_score,
    'false positive rate': false_positive_rate,
    'true positive rate': true_positive_rate,
    'selection rate': selection_rate,
    'count': count}
metric_frame = MetricFrame(metrics=metrics,
                           y_true=labels_test,
                           y_pred=y_pred,
                           sensitive_features=features_test["sex"])
metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
)

# %%
metric_frame.by_group

# %% [markdown]
# Nur GridSearch liefert nicht-randomisierte Ergebnisse und erlaubt auch die Ausgabe von Scores.

# %%
from fairlearn.reductions import ErrorRateParity, GridSearch
mitigator = GridSearch(grid_search.best_estimator_.steps[-1][1], ErrorRateParity())
features_train_tf = Pipeline(grid_search.best_estimator_.steps[:-1]).transform(features_train)
mitigator.fit(features_train_tf, labels_train, sensitive_features=features_train["sex"])

# %% [markdown]
# Konstruktion einer neuen Pipeline mit mitigiertem Classifier

# %%
mitigated_clf = Pipeline(grid_search.best_estimator_.steps[:-1] + [("model", mitigator)])

# %%
metrics = {
    'accuracy': skm.accuracy_score,
    'precision': skm.precision_score,
    'recall': skm.recall_score,
    'false positive rate': false_positive_rate,
    'true positive rate': true_positive_rate,
    'selection rate': selection_rate,
    'count': count}
metric_frame = MetricFrame(metrics=metrics,
                           y_true=labels_test,
                           y_pred=mitigated_clf.predict(features_test),
                           sensitive_features=features_test["sex"])
metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
)

# %%
metric_frame.by_group

# %% [markdown]
# ## Auf allen Daten Trainieren

# %%
sample_weight = compute_sample_weight("balanced", labels)
pipeline.set_params(**grid_search.best_params_)
pipeline.fit(features, labels, **{"clf__sample_weight": sample_weight})

# %%
mitigator = GridSearch(grid_search.best_estimator_.steps[-1][1], ErrorRateParity())
preprocessor = Pipeline(pipeline.steps[:-1])
features_tf = preprocessor.transform(features)
mitigator.fit(features_tf, labels, sensitive_features=features["sex"])
pipeline = Pipeline(preprocessor.steps + [("clf", mitigator)])

# %% [markdown]
# ## Ausgaben speichern
#
# Model speichern

# %%
os.makedirs("../models", exist_ok=True)
with open("../models/model.pkl", "wb") as f:
    cloudpickle.dump(pipeline, f)

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
