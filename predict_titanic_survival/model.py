"""
In diesem Modul wird die Pipeline für das Modell definiert, die trainiert wird und
zur Vorhersage genutzt wird.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from hamlops.data_prep import CustomFeatures

# Beispiel Pipeline
PIPELINE_FULL = Pipeline([
    ("CustomFeatures", CustomFeatures()), # Aufruf des manuellen Feature Engineerings
    ("SelectFeatures", # Auswahl der Features und Umwandlung in numpy Array
        ColumnTransformer([("selected_feature_list", "passthrough", [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
            ])])),
    ("Impute", SimpleImputer()), # Imputieren von Missings mit Mittelwert
    ("Classifier", DecisionTreeClassifier(max_depth=10, min_samples_leaf=10,
        random_state=123)) # DecisionTreeClassifier als Classifier
])
