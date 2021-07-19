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
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder
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

# %% [markdown]
# [Verschiedene Verfahren](https://scikit-learn.org/stable/modules/outlier_detection.html#overview-of-outlier-detection-methods) erlauben die Detektion von Outliern.
#
# Mit Hilfe eines [Isolation Forest](https://scikit-learn.org/stable/modules/outlier_detection.html#isolation-forest) sollen Outlier identifiziert werden. Die Idee hinter dem Isolation Forest ist, alle Datensätze im mehreren Bäumen durch zufällige Splits anhand der Feature in einzelne Blätter aufzuteilen. Outlier benötigen dann durchschnittliche kürzere Pfade um isoliert zu werden.

# %%
od = Pipeline(preprocessor.steps + [("sd", StandardScaler()), ("od", IsolationForest(random_state=12345, contamination=0.02))])
od.fit(train_df.drop(columns=["label"]))

# %% [markdown]
# Hier funktioniert eine [Ein-Klassen-Support-Vector-Maschine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM) besser. Diese sucht nach einer Hyperebene, um alle bekannten Datenpunkte einzugrenzen. Der Outlier Score ist dann die Entfernung zu diese Hyperebene.

# %%
od = Pipeline(preprocessor.steps + [("sd", StandardScaler()), 
#                                    ("pca", PCA(n_components=3)),
                                    ("od", OneClassSVM(kernel="rbf", nu=0.02, gamma=.02))])
od.fit(train_df.drop(columns=["label"]))

# %%
test_df = pd.read_pickle("../data/interim/test_df.pkl")
test_df["label"]=1 # is inliner
outlier_df = pd.read_pickle("../data/interim/outlier_df.pkl")
outlier_df = outlier_df[~outlier_df["fare"].isna()]
outlier_df["label"]=2 # is outlier
inlier_df = train_df
inlier_df["label"]=0 # is outlier
comb_df = pd.concat([inlier_df, outlier_df, test_df])
fig, ax = plt.subplots()
pred = od.decision_function(comb_df.drop(columns=["label"]))
ax.scatter(x=range(inlier_df.shape[0]), y=od.decision_function(inlier_df.drop(columns=["label"])), c="C0",
          label="train")
ax.scatter(x=range(inlier_df.shape[0], test_df.shape[0]+inlier_df.shape[0]),
           y=od.decision_function(test_df.drop(columns=["label"])), c="C1",
          label="test")
ax.scatter(x=range(test_df.shape[0]+inlier_df.shape[0], test_df.shape[0]+inlier_df.shape[0]+outlier_df.shape[0]),
           y=od.decision_function(outlier_df.drop(columns=["label"])), c="C2", label="outlier")
comb_df["label"]
ax.axhline(y=0,c="C1")
ax.set_ylabel("Decision Function")
ax.set_xlabel("Index")
ax.legend()

# %% [markdown]
# Generieren von Datenpunkte, um den Feature Raum abzurastern.

# %%
sibsp, parch, fare, age, embarked, sex, pclass = np.meshgrid(
    [0],# sibsp
    [0],# parch
    np.linspace(0, 100, 50), # fare
    np.linspace(0, 80, 41), # age
    [3],# embarked
    [0],# sex
    [2],# pclass
)

od_short = Pipeline(od.steps[-2:])
# plot the line, the points, and the nearest vectors to the plane
Z = od_short.decision_function(np.c_[
    sibsp.ravel(), 
    parch.ravel(),
    fare.ravel(),
    age.ravel(),
    embarked.ravel(),
    sex.ravel(),
    pclass.ravel(),
])
Z = Z.reshape(age.shape)

# %% [markdown]
# Dartstellung der Entscheidungskontur in der Ebene Alter/Ticketgebühr.

# %%
fig, ax = plt.subplots()
ax.contourf(age[0,0,:, :, 0, 0, 0], fare[0,0,:, :, 0, 0, 0], Z[0,0,:, :, 0, 0, 0], 
            levels=np.linspace(Z[0,0,:, :, 0, 0, 0].min(), 0, 7), cmap=plt.cm.PuBu)
a = ax.contour(age[0,0,:, :, 0, 0, 0], fare[0,0,:, :, 0, 0, 0], Z[0,0,:, :, 0, 0, 0], 
               levels=[0], linewidths=2, colors='darkred')
ax.contourf(age[0,0,:, :, 0, 0, 0], fare[0,0,:, :, 0, 0, 0], Z[0,0,:, :, 0, 0, 0], 
            levels=[0, Z.max()], colors='palevioletred')
s = 40
inlier_prep = preprocessor.transform(inlier_df.drop(columns=["label"])[
    (inlier_df["sibsp"]==0) & (inlier_df["parch"]==0) & (inlier_df["embarked"]=="S") & (inlier_df["sex"]=="female") \
     & (inlier_df["pclass"]==3) 
])
test_prep = preprocessor.transform(test_df.drop(columns=["label"])[
    (test_df["sibsp"]==0) & (test_df["parch"]==0) & (test_df["embarked"]=="S") & (test_df["sex"]=="female") \
     & (test_df["pclass"]==3) 
])
outlier_prep = preprocessor.transform(outlier_df.drop(columns=["label"])[
    (outlier_df["sibsp"]==0) & (outlier_df["parch"]==0) & (outlier_df["embarked"]=="S") & (outlier_df["sex"]=="female") \
     & (outlier_df["pclass"]==3) 
])
b1 = plt.scatter(inlier_prep[:, 3], inlier_prep[:, 2], c='white', s=s, edgecolors='k')
b2 = plt.scatter(test_prep[:, 3], test_prep[:, 2], c='blueviolet', s=s,
                 edgecolors='k')
c = plt.scatter(outlier_prep[:, 3], outlier_prep[:, 3], c='gold', s=s,
                edgecolors='k')
ax.legend([a.collections[0], b1, b2, c],
           ["Gelerne Grenze", "Trainingsdatensätze",
            "Testdatensätze", "Outlier Datensätze"],
           loc="upper left")
ax.set_xlabel("Age")
ax.set_ylabel("Fare")

# %% [markdown]
# Outlier Vorhersage (-1: Outlier)

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
joblib.dump(od, "../models/outlier_detector.pkl")
