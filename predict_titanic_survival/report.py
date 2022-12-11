"""
Module to report performance
"""

import inspect
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.multiclass import type_of_target
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer

def report(gridsearch, features_train, labels_train, \
    features_valid, labels_valid, n_top=3):
    """
    Utility function to report best scores
    http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
    """
    estimator = None
    if hasattr(gridsearch, "cv_results_"):
        results = gridsearch.cv_results_
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: ")
                for key, value in results['params'][candidate].items():
                    print("\t{:40s}: {:20s}".format(key, str(value)))
        estimator = gridsearch.best_estimator_
    else:
        estimator = gridsearch
        print("Parameters: ")
        for key, value in estimator.get_params().items():
            print("\t{:40s}: {:20s}".format(key, str(value)))
    print("\n")
    print("Accuracy train:  {:5.2f} %".format(estimator.score(features_train, labels_train)*100))
    accuracy_test = estimator.score(features_valid, labels_valid)
    print("Accuracy test:   {:5.2f} %".format(accuracy_test*100))
    print("\n")
    y_pred = estimator.predict(features_valid)
    print(classification_report(labels_valid, y_pred))
    print("\n")
    print(confusion_matrix(labels_valid, y_pred))
    print("\n")

    # Compute predicted probabilities: y_pred_prob
    if hasattr(estimator, "decision_function"):
        #e.g. svm
        y_pred_prob = estimator.decision_function(features_valid)
    else: 
        y_pred_prob = estimator.predict_proba(features_valid)[:, 1]

    scores = {"accuracy": accuracy_test}

    target_type = type_of_target(labels_valid)
    if target_type == "binary":
        # Generate values for ROC curve 
        fpr, tpr, _ = roc_curve(labels_valid, y_pred_prob)
        auc_value = auc(fpr, tpr)
        print("AUC:", auc_value)
        scores["auc"] = auc_value
    
    return scores

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
                if hasattr(transformer, "feature_names_in_"):
                    if ((not isinstance(column, slice)) and all(isinstance(col, str) for col in column)):
                        feature_names.extend(column)
                    else:
                        feature_names.extend(transformer.feature_names_in_[column])
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