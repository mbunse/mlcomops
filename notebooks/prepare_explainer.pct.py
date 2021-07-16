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
# dvc run -n prepare_explainer --force -d ../models/model.pkl -d ../data/interim/train_df.pkl -d ../data/interim/test_df.pkl -d ../models/feat_names.json -o ../models/explainer.pkl -w notebooks python prepare_explainer.pct.py
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

# %%
list(enumerate(feat_names))

# %% [markdown]
# ## Explainer erzeugen

# %%
explainer = LimeTabularExplainer(train_df_tf, feature_names=feat_names, 
                                 class_names=["died", "survived"], discretize_continuous=True,
                                categorical_features=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                categorical_names={
                                    0: ["1st class", "2nd class", "3rd class"],
                                    5: ["yes", "no"],
                                    6: ["yes", "no"],
                                    7: ["yes", "no"],
                                    8: ["yes", "no"],
                                    9: ["yes", "no"],
                                    10: ["yes", "no"],
                                    11: ["yes", "no"],
                                    12: ["yes", "no"],
                                    13: ["yes", "no"],
                                    14: ["yes", "no"],
                                    15: ["yes", "no"],
                                    16: ["yes", "no"],
                                })

# %% [markdown]
# Explainer testen

# %%
instance = preprocessor.transform(test_df.drop(columns=["label"]))[0]
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
