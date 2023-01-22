#!/bin/python
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# # Model Training
# The model is persisted to `model/model.pkl`.
#
# Create dvc pipeline:
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
from predict_titanic_survival.report import report, get_feature_names

# %% [markdown] hideOutput=false hideCode=true
# ## Read Data

# %%
filepath = "../data/interim/train_df.pkl"

data_df = pd.read_pickle(filepath)
labels = data_df["label"].copy()
features = data_df.drop(columns=["label"]).copy()

# %%
features_train, features_valid, labels_train, labels_valid = train_test_split(
    features, labels, random_state=12345, stratify=labels)
features_train.shape, features_valid.shape, labels_train.shape, labels_valid.shape

# %% [markdown]
# ## Define model pipeline
#
# Only the numeric columns `sibsp`, `parch`, `fare` and `age` and the categorical columns `embarked`, `sex` and `pclass` are used.
#
# The categorical columns are coded ordinally, since this is advantageous e.g. for the treatment in the context of the drift detection later and decision tree-based classifiers have no difficulties with this either.
#
# Before the data is run into the classifier, missing values are imputed in the pipeline.

# %%
pipeline = Pipeline([
    ("select_feat", ColumnTransformer([
        ("select_num_cols", "passthrough", ["sibsp", "parch", "fare", "age"]),
        ("encode_str_cols", Pipeline([
            ("replace_nan", SimpleImputer(strategy="constant", fill_value="Missing")), 
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=4))
        ]), ["embarked", "sex", "pclass"])
    ], remainder="drop")),
    ("impute", FeatureUnion([
        ("imputed", KNNImputer()),
        ("miss_indicator", MissingIndicator()),
    ])),
    ("clf", GradientBoostingClassifier(random_state=1234))
])

# %% [markdown]
# The preprocessor pipeline is now fitted and the data is transformed on a test basis.

# %%
preprocessor = Pipeline(pipeline.steps[:-1])
X_transformed = preprocessor.fit_transform(features_train, labels_train)
X_transformed.shape

# %% [markdown]
# Now the names of the preprocessed columns can be derived.

# %%
feat_names = get_feature_names(pipeline)
assert len(feat_names)==X_transformed.shape[1]
feat_names

# %% [markdown]
# What are the parameters of the pipeline?

# %%
list(pipeline.get_params().keys())

# %% [markdown]
# ## Experiment Tracking
#
# Experiment tracking woth [mlflow](https://mlflow.org/).

# %%
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.sklearn.autolog(log_model_signatures=False, log_models=False)
mlflow.set_experiment('New experiment')

# %% [markdown]
# The run data is stored directly in the S3 bucket stored in the tracking server. Therefore, the corresponding access data is required here.

# %%
os.environ["AWS_ACCESS_KEY_ID"]='minio-access-key'
os.environ["AWS_SECRET_ACCESS_KEY"]='minio-secret-key'
os.environ["MLFLOW_S3_ENDPOINT_URL"]='http://localhost:9000'

# %% [markdown]
# Perform GridSearch for hyperparameter optimization.

# %%
param_grid = {
    "clf__max_depth": [2, 3],
    "clf__min_samples_leaf": [5, 20]
}
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=4, n_jobs=4, scoring="accuracy")
sample_weight = compute_sample_weight("balanced", labels_train)

with mlflow.start_run() as run:
    # Grid-Search unter Berücksichtigung der Sample-Weights durchführen
    grid_search.fit(features_train, labels_train, **{"clf__sample_weight": sample_weight})

# %% [markdown]
# The results can now be viewed in [MLFlow](http://localhost:5000)

# %% [markdown]
# ## Calculate Fairness Metrics

# %%
y_pred = grid_search.best_estimator_.predict(features_valid)

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
                           y_true=labels_valid,
                           y_pred=y_pred,
                           sensitive_features=features_valid["sex"])
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
# The idea behind [fairlearn](https://fairlearn.org/) is based on [Agarwal et al.](https://arxiv.org/pdf/1803.02453.pdf). In short, a Lagrange multiplier is used in a grid search to adjust the weights of individual data points in such a way that the remaining deviations from fairness are weighted more. The classifier that provides the best trade-off between fairness and performance is used. [GridSearch](https://fairlearn.org/v0.5.0/api_reference/fairlearn.reductions.html#fairlearn.reductions.GridSearch) provides non-randomized results and also allows the output of scores.

# %%
from fairlearn.reductions import ErrorRateParity, GridSearch
np.random.seed(seed=12345)
mitigator = GridSearch(grid_search.best_estimator_.steps[-1][1], ErrorRateParity())
features_train_tf = Pipeline(grid_search.best_estimator_.steps[:-1]).transform(features_train)
mlflow.autolog()
with mlflow.start_run() as fairness_run:
    mitigator.fit(features_train_tf, labels_train, sensitive_features=features_train["sex"])
    
    # Konstruktion einer neuen Pipeline mit mitigiertem Classifier
    mitigated_clf = Pipeline(grid_search.best_estimator_.steps[:-1] + [("model", mitigator)])
    
    # Metriken in MLFlow loggen
    y_pred = mitigated_clf.predict(features_valid)
    prob_pred = mitigated_clf.predict_proba(features_valid)
    scores = {
        "test_accuracy_score": skm.accuracy_score(labels_valid, y_pred),
        "test_f1_score": skm.f1_score(labels_valid, y_pred),
        "test_precision_score": skm.precision_score(labels_valid, y_pred),
        "test_roc_auc_score": skm.roc_auc_score(labels_valid, y_pred),
    }
    mlflow.log_metrics(scores)

# %% [markdown]
# ### Mitigation details
#
# The following grid is used.

# %%
mitigator.lambda_vecs_

# %% [markdown]
# The following lambda value provided the best classifier according to the tradeoff.

# %%
mitigator.best_idx_


# %% [markdown]
# The tradeoff takes the following values:

# %%
def loss_fct(i):
    return mitigator.objective_weight * mitigator.objectives_[i] + \
        mitigator.constraint_weight * mitigator.gammas_[i].max()
{idx: loss_fct(idx) for idx in mitigator.lambda_vecs_.columns}

# %% [markdown]
# ## Metrics after Mitigation

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
                           y_true=labels_valid,
                           y_pred=mitigated_clf.predict(features_valid),
                           sensitive_features=features_valid["sex"])
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
# ## Train on all Data

# %%
mlflow.sklearn.autolog(disable=True)

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
# ## Save Output
#
# Model speichern

# %%
os.makedirs("../models", exist_ok=True)
with open("../models/model.pkl", "wb") as f:
    cloudpickle.dump(pipeline, f)

# %% [markdown]
# Save metrics

# %%
with open("../models/score.json", "w") as f:
    json.dump(scores, f)

# %% [markdown]
# Save feature names

# %%
with open("../models/feat_names.json", "w") as f:
    json.dump(feat_names, f)
