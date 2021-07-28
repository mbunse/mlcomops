# MLOps

## Voraussetzungen
* Conda Installation (z.B. [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
* Base Umgebung mit Jupyter, [nb_conda](https://anaconda.org/conda-forge/nb_conda) und [jupytext](https://jupytext.readthedocs.io/en/latest/install.html)
## Einrichtung

Nach dem Checkout zunächst die conda Umgebung erstellen und aktivieren:
```
git clone https://github.com/datanizing/datascienceday.git
cd datascienceday
cd 08_MLOps
conda env create -f environment.yml
conda activate mlops
```

Unter Linux Systemen die Berechtigungen für die gemounteten Volumes anpassen:
```
chmod -R o+w docker-compose/
```

Damit nun die DVC Dateien getrackt werden, den untersten markierten Abschnitt der [.gitignore] auskommentieren.

## Benötigten Komponenten bereitstellen

```
docker-compose build
docker-compose up
```

In [Minio](http://localhost:9000) ein Bucket titanic anlegen.

## DVC initialisieren

```
dvc init
dvc remote add -d minio s3://titanic/dvcrepo
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify --local minio access_key_id minio-access-key
dvc remote modify --local minio secret_access_key minio-secret-key
```
## Datensatz

[Titanic Datensatz von OpenML](https://www.openml.org/d/40945)

Daten laden als dvc Stage anlegen und ausführen:
```
dvc run -n load_data --force -o ../data/interim/train_df.pkl -o ../data/interim/test_df.pkl -o ../data/interim/outlier_df.pkl -d load_data.pct.py -w notebooks python load_data.pct.py
```



## Starten der API:

```
python app.py
```

## API erzeugen

Während die Model API läuft, folgendes ausführen:
```
openapi-python-client generate --url http://127.0.0.1:8080/openapi.json
```


## Setup aus Prometheus, Grafana und Model API starten

Voraussetzung: 
* lokale [Docker](https://docs.docker.com/get-docker/) Installation

```
chmod o+x docker-compose/modelapi/data
docker-compose build
docker-compose -f docker-compose.yml -f docker-compose.modelapi.yml up
```
API: http://localhost:8080/docs

## Monitoring

* [Grafana öffnen](http://locaadminlhost:3000)
* [Prometheus öffen](http://localhost:9090)

![Dashboard](images/dashboard.png)
Inspired by [Jeremy Jordan
A simple solution for monitoring ML systems.
](https://www.jeremyjordan.me/ml-monitoring/)

## Referenzen:
* [fastAPI](https://fastapi.tiangolo.com/)
* [pydantic]()
* [dvc](https://dvc.org/)
* [openapi-python-client](https://github.com/openapi-generators/openapi-python-client)
* [prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator)
* [Docker](https://docs.docker.com/get-docker/)
* [OpenShift](https://www.openshift.com/)
* [minishift](https://docs.okd.io/3.11/minishift/getting-started/index.html)
* [Grafana](https://grafana.com/)
* [Prometheus](https://prometheus.io/)
