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
# # Prepare Explainer
#
# ```
# dvc run -n outlier_model --force -w notebooks -d ../data/interim/train_df.pkl -d ../data/interim/test_df.pkl -d ../models/feat_names.json -d ../models/model.pkl -d ../data/interim/outlier_df.pkl -o ../models/explainer.pkl python prepare_explainer.pct.py
# ```

# %%
import joblib
import json
import pandas as pd
import cloudpickle
from sklearn.pipeline import Pipeline

from lime.lime_tabular import LimeTabularExplainer

# %% [markdown]
# ## Model und Daten laden

# %%
pipeline = joblib.load("../models/model.pkl")
clf = pipeline.steps[-1][1]
train_df = pd.read_pickle("../data/interim/train_df.pkl")
test_df = pd.read_pickle("../data/interim/test_df.pkl")
feat_names = json.load(open("../models/feat_names.json", "r"))

# %% [markdown]
# Trainingsdaten transformieren

# %%
preprocessor = Pipeline(pipeline.steps[:-1])
train_df_tf = preprocessor.transform(train_df.drop(columns=["label"]))

# %% [markdown]
# Folgende Feature gibt es

# %%
list(enumerate(feat_names))

# %% [markdown]
# Die Werte der kategoriellen Feature sind wie folgt zu mappen.

# %%
preprocessor.steps[0][1].transformers_[1][1].steps[1][1].categories_

# %% [markdown]
# ## Create explainer
#
# For explainability, [lime](https://github.com/marcotcr/lime) is used. [Lime](https://arxiv.org/abs/1602.04938) fitted an explainable algorithm ([Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)) to the model prediction for samples similar to the instance in question.

# %%
explainer = LimeTabularExplainer(train_df_tf, feature_names=feat_names, 
                                 class_names=["died", "survived"], discretize_continuous=True,
                                categorical_features=[4, 5, 6],
                                categorical_names={
                                    6: ["1st class", "2nd class", "3rd class"],
                                    4: ['C', 'Missing', 'Q', 'S'],
                                    5: ['female', 'male'],
                                })

# %% [markdown]
# The explanation using the example of a test data set

# %%
instance = preprocessor.transform(test_df.drop(columns=["label"]))[0]
explanation = explainer.explain_instance(instance, clf.predict_proba)

# %%
explanation.show_in_notebook()

# %%
test_df.iloc[0]

# %%
{value: explain for value, explain in explanation.as_list()}

# %% [markdown]
# ## Save

# %%
with open("../models/explainer.pkl", "wb") as f:
    cloudpickle.dump(explainer, f)
