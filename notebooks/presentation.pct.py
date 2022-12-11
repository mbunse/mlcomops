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

# %% [markdown] slideshow={"slide_type": "slide"}
# # MLOps
#
# ![MLOps](../images/mlops_venn.jpg)

# %% [markdown] slideshow={"slide_type": "subslide"}
# # MLOps
# ![Technical debt](../images/tech_debt.png)
#
# [Hidden Technical Debt in Machine Learning Systems by D. Sculley et al. from 2015](https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Poll
# https://strawpoll.com/polls/NoZrLa3aXn3

# %% hideCode=true
from IPython.display import IFrame
IFrame('https://strawpoll.com/embed/NoZrLa3aXn3', width=700, height=350)

# %% [markdown] slideshow={"slide_type": "slide"}
# # Before the project starts
#

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Data Protection
# Clarify legal basis for your purpose ([Art. 6 GDPR](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679&from=DE#d1e1888-1-1)):  
# * data subject has given consent to the processing
# * fulfillment of a contract
# * legal requirements
# * vital interests of the data subjects
# * performance of a task in the public interest
# * legitimate interest
#
# Data protection impact assessment usually required, as "automated processing" (Art 35(3) GDPR). The lists of the state data protection authorities must also be taken into account (e.g. [BayLDA: List of processing activities for which a DSFA must be performed](https://www.lda.bayern.de/media/dsfa_muss_liste_dsk_de.pdf)).

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Processing of employee-related data
#
# The workers' council must be informed about the processing of employee data within the framework of co-determination ([Section 87 (1) No. 6 German Works Constitution Act](https://www.gesetze-im-internet.de/betrvg/__87.html)).

# %% [markdown] slideshow={"slide_type": "skip"}
# # Implementation of regulatory requirements in machine learning projects
# https://github.com/mbunse/mlcomops

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## GDPR Requirements
# [Position paper of the Data Protection Conference on recommended technical and
# organizational measures for the development and operation
# of AI systems](https://www.datenschutzkonferenz-online.de/media/en/20191106_positionspapier_kuenstliche_intelligenz.pdf) as of Nov. 2019.
#
# Required in the position paper (excerpts):
# * Documentation of the selection of the AI process (balancing traceability and required power).
# * Preservation of availability of raw and training data.
# * Prevention of unauthorized manipulation of AI components
# * Possibility for data subjects to obtain information on how decisions and predictions were made
# * Monitoring of the behavior of the AI component
# * Regular testing of the AI component for discrimination and other undesirable behavior
# * Regular testing of the quality of the AI system and its AI components on the basis of operational data.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## EU regulation
#
# [Proposal for a Regulation laying down harmonised rules on artificial intelligence](https://digital-strategy.ec.europa.eu/en/library/proposal-regulation-laying-down-harmonised-rules-artificial-intelligence)
#
# <img alt="EU risk categories" src="https://commission.europa.eu/sites/default/files/styles/oe_theme_medium_no_crop/public/2021-04/ai_pyramid_visual-01.jpg" width=700/>
#
# High risk e.g.
# * AI in road traffic
# * **Credit scoring**

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## BaFin
# ![Bafin](../images/bafin_ml_principles.png)
#
# New MaRisk (Minimal requirements for risk management) proposal (Sep. 26 2022) from Federal Financial Supervisory Authority (BaFin) includes regulations for models and artifical intelligence used in risk management processes.
#
# [Supervisory Principles for Big Data and AI from 6/15/2021](https://www.bafin.de/SharedDocs/Downloads/DE/Aufsichtsrecht/dl_Prinzipienpapier_BDAI.html)

# %% [markdown] slideshow={"slide_type": "subslide"}
# Aspects to be highlighted:
#
# * __Reproducibility__: versioning of data and code:
#     * Maintaining availability of raw and training data.
# * __Experiment Tracking__:
#     * Documentation of the selection of the AI procedure.
#     * Evaluation of the selected AI procedure with respect to alternative, more explainable AI procedures.
# * __Fairness__:
#     * Periodic testing of the AI component for discrimination and other undesirable behavior.
# * __Model Explainability__:
#     * Ability to provide information to affected parties on how decisions and predictions were made.
# * __Monitoring__:
#     * monitoring of the behavior of the AI component

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Poll
# https://strawpoll.com/daprodsdy

# %% hideCode=true
from IPython.display import IFrame
IFrame('https://strawpoll.com/embed/daprodsdy', width=700, height=350)

# %% [markdown] slideshow={"slide_type": "slide"}
# # Reproducibility

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Data set
#
# ![Hey Rose, according to this ML model I must stay in water and freeze](https://www.nagarajbhat.com/post/predicting-titanic-survival/featured.jpg)

# %% [markdown] slideshow={"slide_type": "fragment"}
# [Load data](load_data.pct.py)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Poll
# https://strawpoll.com/kc8pxhafz

# %% hideCode=true
from IPython.display import IFrame
IFrame('https://strawpoll.com/embed/kc8pxhafz', width=700, height=350)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Data Versioning, 
#
# e.g. with [DVC](https://dvc.org/)
# ![DVC](../images/data_code_versioning.png)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## DVC Data
#
# [DVC Remote with Minio](http://localhost:9000/minio/titanic/)

# %% slideshow={"slide_type": "fragment"}
# ! cd .. & dvc pull

# %% slideshow={"slide_type": "fragment"}
# ! cd .. & dvc repro

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## DVC Data

# %%
# ! cd .. & dvc push

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Set all random seeds
# cf. ["Confusion about R-value calculation of the RKI"](https://www.spiegel.de/politik/deutschland/robert-koch-institut-und-der-r-wert-ende-april-verwirrung-ueber-berechnung-a-264a8d9c-454f-499a-b729-e4b537688b72)
# > Within this simulation, random numbers are drawn that will result slightly different each time the program is run and therefore cannot be exactly reproduced.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Experiment Tracking
#
# Experiment Tracking with [MLflow](https://mlflow.org/)
#
# [Modell Training](train.pct.py)
#
# [MLFlow](http://localhost:5000)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Experiment Tracking with [MLflow](https://mlflow.org/)
#
# ```python
# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.sklearn.autolog(log_model_signatures=False, log_models=False)
# mlflow.set_experiment('New experiment')
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Fairness in Machine Learning Projects
#
# Prohibition of discrimination according to [§ 19 AGG](https://www.gesetze-im-internet.de/agg/__19.html).
#
# [Model Training](train.pct.py)
#
# https://towardsdatascience.com/real-life-examples-of-discriminating-artificial-intelligence-cae395a90070
#
# ![Equity](https://miro.medium.com/max/408/1*hntbZ9h50ql9dxoP0FQfVQ.png)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Fairlearn
#
# The idea behind [fairlearn](https://fairlearn.org/) is based on [Agarwal et al.](https://arxiv.org/pdf/1803.02453.pdf). In short, a Lagrange multiplier is used in a grid search to adjust the weights of individual data points in such a way that the remaining deviations from fairness are weighted more. The classifier that provides the best trade-off between fairness and performance is used. [GridSearch](https://fairlearn.org/v0.5.0/api_reference/fairlearn.reductions.html#fairlearn.reductions.GridSearch) provides non-randomized results and also allows the output of scores.
#
# ```python
# from fairlearn.reductions import ErrorRateParity, GridSearch
# mitigator = GridSearch(classifier, ErrorRateParity())
# mitigator.fit(features_train_tf, labels_train, sensitive_features=features_train["sex"])
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Explainability
#
# [Explainer](prepare_explainer.pct.py)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Deploying the model
# ![deploy](../images/deploy.png)
# [Model API](http://localhost:8080/docs)

# %% [markdown] slideshow={"slide_type": "slide"}
# # Monitoring
#
# [Model API](http://localhost:8080/docs)
#
# [Metrics Endpoint](http://localhost:8080/metrics)
#
# [Grafana](http://localhost:3000)
#
# [Call API](call_api.pct.py)

# %% [markdown] slideshow={"slide_type": "subslide"}
# # Monitoring
#
# ![dashboard showing distriubtions of models scores, outlier scores, labels and drifts over time](../images/dashboard.png)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### What to watch for?
#
# * online vs. offline scores
# * evaluation metrics
# * score distributions
# * feature distribtions

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Outlier Detection
# <img style="height:600px;" src="https://i.stack.imgur.com/3Ab7e.jpg" alt="Extrapolation" />
#
# [Outlier Detector](outlier_detector.pct.py)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Drift detection
#
# [Drift detection](drift_detector.pct.py)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Outlier Detection
# [Call API with outliers](call_api.pct.py)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Automation

# %% slideshow={"slide_type": "-"} hideCode=true language="html"
# <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Machine learning pipelines <a href="https://t.co/5FpG3HrdW0">pic.twitter.com/5FpG3HrdW0</a></p>&mdash; AI Memes for Artificially Intelligent Teens (@ai_memes) <a href="https://twitter.com/ai_memes/status/1382374419666976771?ref_src=twsrc%5Etfw">April 14, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## What else?
# * Data Science development environment
# * Pull Requests
# * Test Automation
#     * Unit Tests
#     * Integration tests
# * Scaling (e.g. with Kubernetes)
# * Staging
# * CI/CD
# * security
# * ...
