"""
Module to report performance
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.multiclass import type_of_target

def report(gridsearch, features_train, labels_train, \
    features_test, labels_test, n_top=3):
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
    accuracy_test = estimator.score(features_test, labels_test)
    print("Accuracy test:   {:5.2f} %".format(accuracy_test*100))
    print("\n")
    y_pred = estimator.predict(features_test)
    print(classification_report(labels_test, y_pred))
    print("\n")
    print(confusion_matrix(labels_test, y_pred))
    print("\n")

    # Compute predicted probabilities: y_pred_prob
    if hasattr(estimator, "decision_function"):
        #e.g. svm
        y_pred_prob = estimator.decision_function(features_test)
    else: 
        y_pred_prob = estimator.predict_proba(features_test)[:, 1]

    scores = {"accuracy": accuracy_test}

    target_type = type_of_target(labels_test)
    if target_type == "binary":
        # Generate values for ROC curve 
        fpr, tpr, _ = roc_curve(labels_test, y_pred_prob)
        auc_value = auc(fpr, tpr)
        print("AUC:", auc_value)
        scores["auc"] = auc_value
    
    return scores
