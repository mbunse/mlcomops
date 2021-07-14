"""
Modul mit Tranformatoren für sklearn Pipelines.

"""

from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatures(BaseEstimator, TransformerMixin):
    """
    Klasse für manuelles Feature Engineering innerhlab einer Pipeline.
    """
    def __init__(self):
        """
        Hier müssen ggf. Eigenschaften ergänzt werden.
        """
    # pylint: disable=unused-argument,missing-docstring
    def fit(self, X, y=None):
        return self

    # pylint: disable=no-self-use
    def transform(self, X):
        """
        Transform Methode, die einen pandas DataFrame 
        als Eingabe erwartet und einen DataFrame
        mit den ergänzten Features zurückgibt.
        """
        X = X.copy()
        
        X["sepal rectangle (cm2)"] = X["sepal length (cm)"] * X["sepal width (cm)"]

        return X
