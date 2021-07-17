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
# # Explainer vorbereiten
# ```
# dvc run -n prepare_explainer --force -d ../models/model.pkl -d ../data/interim/valid_df.pkl -d ../data/interim/test_df.pkl -d ../models/feat_names.json -o ../models/explainer.pkl -w notebooks python prepare_explainer.pct.py
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
valid_df = pd.read_pickle("../data/interim/valid_df.pkl")
feat_names = json.load(open("../models/feat_names.json", "r"))

# %% [markdown]
# Trainingsdaten transformieren

# %%
preprocessor = Pipeline(pipeline.steps[:-1])
train_df_tf = preprocessor.transform(train_df.drop(columns=["label"]))

# %%
list(enumerate(feat_names))

# %%
preprocessor.steps[0][1].transformers_[1][1].steps[1][1].categories_

# %% [markdown]
# ## Explainer erzeugen

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
# Explainer testen

# %%
instance = preprocessor.transform(valid_df.drop(columns=["label"]))[0]
explanation = explainer.explain_instance(instance, clf.predict_proba)

# %%
explanation.show_in_notebook()

# %%
{value: explain for value, explain in explanation.as_list()}

# %% [markdown]
# ## Speichern

# %%
with open("../models/explainer.pkl", "wb") as f:
    cloudpickle.dump(explainer, f)
