# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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

# %% [markdown] slideshow={"slide_type": "slide"}
# # ComMLOps
#
# ![MLOps](https://nuernberg.digital/uploads/tx_seminars/praesentation2.jpg)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Disclaimer
#
# * Meine private Ansicht auf Basis meiner Erfahrung
# * Ich bin kein Jurist
# * Kein Anspruch auf Vollständigkeit der rechtlichen & regulatorischen Anforderungen
#
# ![Disclaimer](https://i.imgflip.com/5gs3cu.jpg)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Umfrage
# https://strawpoll.com/o29kygyz2

# %% hideCode=true
from IPython.display import IFrame
IFrame('https://strawpoll.com/embed/o29kygyz2', width=700, height=350)

# %% [markdown] slideshow={"slide_type": "slide"}
# # Bevor das Projekt startet
#

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Datenschutz klären
# Rechtsgrundlage für Verarbeitungszweck klären ([Art. 6 DSGVO](https://dsgvo-gesetz.de/art-6-dsgvo/)):  
# * Einwilligung
# * Erfüllung eines Vertrages
# * rechtliche Vorgaben
# * lebenswichtige Interessen der betroffenen Personen
# * Wahrnehmung einer Aufgabe im öffentlichen Interesse
# * berechtigtes Interesse
#
# I.d.R. Datenschutzfolgeabschätzung erforderlich, da "automatisierte Verarbeitung" \([Art 35 Absatz 3 DSGVO](https://dsgvo-gesetz.de/art-35-dsgvo/)\). Auch die Listen der Landesdatenschutzbehörden sind zu beachten (z.B. [BayLDA: Liste der Verarbeitungstätigkeiten, für die eine DSFA durchzuführen ist](https://www.lda.bayern.de/media/dsfa_muss_liste_dsk_de.pdf))

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Verarbeitung von Mitarbeiterbezogenen Daten
#
# Im Rahmen der betrieblichen Mitbestimmung ([§ 87 Absatz 1 Nr. 6](https://www.gesetze-im-internet.de/betrvg/__87.html)) ist der Betriebsrat über die Verarbeitung von Mitarbeiterdaten zu informieren.

# %% [markdown] slideshow={"slide_type": "slide"}
# # Umsetzung
# https://github.com/mbunse/mlcomops

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Anforderungen
# [Positionspapier der Datenschutz-Koferenz zu empfohlenen technischen und
# organisatorischen Maßnahmen bei der Entwicklung und dem Betrieb
# von KI-Systemen](https://www.datenschutzkonferenz-online.de/media/en/20191106_positionspapier_kuenstliche_intelligenz.pdf) von Nov. 2019.
#
# Im Positionspapier gefordert (auszugsweise):
# * Dokumentation der Auswahl des KI-Verfahrens (Abwägung zwischen Nachvollziehbarkeit und benötigter Mächtigkeit)
# * Wahrung der Verfügbarkeit von Roh- und Trainingsdaten
# * Verhinderung von unbefugten Manipulationen an KI-Komponenten
# * Auskunftsmöglichkeit für Betroffene zum Zustandekommen von Entscheidungen und Prognosen
# * Überwachung des Verhaltens der KI-Komponente
# * Regelmäßige Prüfung der KI-Komponente auf Diskriminierungen und anderes unerwünschtes Verhalten
# * Regelmäßige Prüfung der Güte des KI-Systems und seiner KI-Komponenten auf Basis der Betriebsdaten

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## EU Regulierung
#
# [Proposal for a Regulation laying down harmonised rules on artificial intelligence](https://digital-strategy.ec.europa.eu/en/library/proposal-regulation-laying-down-harmonised-rules-artificial-intelligence)
#
# <img alt="EU risk categories" src="https://ec.europa.eu/info/sites/default/files/ai_pyramid_visual-01.jpg" style="height:400px;"/>
#
# Hochrisiko z.B.
# * KI im Straßenverkehr
# * Kreditwürdigkeitsprüfung

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## BaFin
# ![Bafin](../images/bafin_ml_risikomodelle.png)
#
# [Prinzipienpapier Big Data und KI vom 15.6.2021](https://www.bafin.de/SharedDocs/Downloads/DE/Aufsichtsrecht/dl_Prinzipienpapier_BDAI.html)

# %% [markdown] slideshow={"slide_type": "subslide"}
# Aspekte, die beleuchtet werden sollen:
#
# * __Reproduzierbarkeit__: Versionierung von Daten und Code:
#     * Wahrung der Verfügbarkeit von Roh- und Trainingsdaten
# * __Experiment Tracking__:
#     * Dokumentation der Auswahl des KI-Verfahrens
#     * Evaluation des ausgewählten KI-Verfahrens bezüglich alternativer, erklärbarerer KI-Verfahren
# * __Fairness__:
#     * Regelmäßige Prüfung der KI-Komponente auf Diskriminierungen und anderes unerwünschtes Verhalten
# * __Modell Erklärbarkeit__:
#     * Auskunftsmöglichkeit für Betroffene zum Zustandekommen von Entscheidungen und Prognosen
# * __Monitoring__:
#     * Überwachung des Verhaltens der KI-Komponente

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Umfrage
# https://strawpoll.com/kezy3yudu
# <!--![QR1](../images/qr_survey_1.png)-->

# %% hideCode=true
from IPython.display import IFrame
IFrame('https://strawpoll.com/embed/kezy3yudu', width=700, height=350)

# %% [markdown] slideshow={"slide_type": "slide"}
# # Reproduzierbarkeit

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Datensatz
#
# ![Hey Rose, according to this ML model I must stay in water and freeze](https://www.nagarajbhat.com/post/predicting-titanic-survival/featured.jpg)

# %% [markdown] slideshow={"slide_type": "fragment"}
# [Daten laden](load_data.pct.py)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Daten versionieren, 
#
# z.B. mit [DVC](https://dvc.org/)
# ![DVC](../images/data_code_versioning.png)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## DVC Daten
#
# [DVC Remote in Minio](http://localhost:9000/minio/titanic/)

# %% slideshow={"slide_type": "fragment"}
# ! cd .. & dvc pull

# %% slideshow={"slide_type": "fragment"}
# ! cd .. & dvc repro

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## DVC Daten

# %%
# ! cd .. & dvc push

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Alle Random Seeds setzen
# vgl. ["Verwirrung über R-Wert-Berechnung des RKI"](https://www.spiegel.de/politik/deutschland/robert-koch-institut-und-der-r-wert-ende-april-verwirrung-ueber-berechnung-a-264a8d9c-454f-499a-b729-e4b537688b72)
# > Innerhalb dieser Simulation werden Zufallszahlen gezogen, die sich bei jedem Lauf des Programms leicht unterschiedlich ergeben und daher nicht exakt reproduzierbar sind.

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Experiment Tracking
#
# Experiment Tracking z.B. mit [MLflow](https://mlflow.org/)
#
# [Modell Training](train.pct.py)
#
# [MLFlow](http://localhost:5000)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Fairness in Machine Learning Projekten
#
# Benachteiligungsverbot gemäß [§ 19 AGG](https://www.gesetze-im-internet.de/agg/__19.html).
#
# https://towardsdatascience.com/real-life-examples-of-discriminating-artificial-intelligence-cae395a90070
#
# ![Equity](https://miro.medium.com/max/408/1*hntbZ9h50ql9dxoP0FQfVQ.png)
#
# [Modell Training](train.pct.py)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Erklärbarkeit
#
# [Explainer](prepare_explainer.pct.py)

# %% [markdown] slideshow={"slide_type": "slide"}
# # Monitoring
#
# [Model API](http://localhost:8080/docs)
#
# [Metrics Endpoint](http://localhost:8080/metrics)
#
# [Grafana](http://localhost:3000)
#
# [API aufrufen](call_api.pct.py)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Outlier Erkennung
# <img style="height:600px;" src="https://i.stack.imgur.com/3Ab7e.jpg" alt="Extrapolation" />
#
# [Outlier Detector](outlier_detector.pct.py)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Outlier Erkennung
# [API mit Outliern aufrufen](call_api.pct.py)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Automatisierung

# %% slideshow={"slide_type": "-"} hideCode=true language="html"
# <blockquote class="twitter-tweet"><p lang="en" dir="ltr">Machine learning pipelines <a href="https://t.co/5FpG3HrdW0">pic.twitter.com/5FpG3HrdW0</a></p>&mdash; AI Memes for Artificially Intelligent Teens (@ai_memes) <a href="https://twitter.com/ai_memes/status/1382374419666976771?ref_src=twsrc%5Etfw">April 14, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Was gibt es noch?
# * Data Science Entwicklungsumgebung
# * Pull Requests
# * Test Automatisierung
#     * Unit-Tests
#     * Intergrations-Tests
# * Skalierung (z.B. mit Kubernetes)
# * Staging
# * CI/CD
# * Security
# * ...
