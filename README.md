# ComMLOps

[PDF Version of the presentation](presentation/presentation.pdf)

## Requirements
* Conda installation (e.g. [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
* Base environment with Jupyter, [nb_conda](https://anaconda.org/conda-forge/nb_conda) and [jupytext](https://jupytext.readthedocs.io/en/latest/install.html)
## Setup

After checkout, first create and activate the conda environment:
```
git clone https://github.com/datanizing/datascienceday.git
cd datascienceday
cd 08_MLOps
conda env create -f environment.yml
conda activate mlops
```

On Linux systems, adjust the permissions for the mounted volumes:
```
chmod -R o+w docker-compose/
```

Now, in order for the DVC files to be tracked, comment out the lowest highlighted section of the [.gitignore].

## Deploy required components

```
docker-compose build
docker-compose up
```

Create a bucket titanic in [Minio](http://localhost:9000).

## Initialize DVC

```
dvc init
dvc remote add -d minio s3://titanic/dvcrepo
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify --local minio access_key_id minio-access-key
dvc remote modify --local minio secret_access_key minio-secret-key
```

## record

[Titanic dataset from OpenML](https://www.openml.org/d/40945)

Load data as dvc Stage create and run:
```
dvc run -n load_data --force -o ../data/interim/train_df.pkl -o ../data/interim/test_df.pkl -o ../data/interim/outlier_df.pkl -d load_data.pct.py -w notebooks python load_data.pct.py
```

## Start the API:

```
python app.py
```

## Create API

While the Model API is running, run the following:
```
openapi-python-client generate --url http://127.0.0.1:8080/openapi.json
```


## Start setup from Prometheus, Grafana and Model API.

Prerequisite: 
* local [Docker](https://docs.docker.com/get-docker/) installation.

```
chmod o+x docker-compose/modelapi/data
docker-compose build
docker build -t modelapi --network="host" --build-arg AWS_ACCESS_KEY_ID="minio-access-key" --build-arg AWS_SECRET_ACCESS_KEY="minio-secret-key" .
docker-compose -f docker-compose.yml -f docker-compose.modelapi.yml up
```
API: http://localhost:8080/docs

## Debugging container
```
docker exec -it mlcomops_openldap_1 /bin/sh
```
## Ldap
```
ldapsearch -H ldap://localhost:1389 -x -b 'dc=example,dc=org'
ldapsearch -H ldap://localhost:1389 -x -b 'ou=users,dc=example,dc=org' '(&(cn=readers)(objectClass=groupOfNames))'
ldapsearch -H ldap://localhost:1389 -x -b 'ou=users,dc=example,dc=org' -s one -a find '(&(cn=readers)(objectClass=groupOfNames))' uid member cn objectclass
ldapsearch -H ldap://localhost:1389 -x -b 'ou=users,dc=example,dc=org' '(&(objectClass=groupOfNames))'
ldapsearch -H ldap://localhost:1389 -x -b 'ou=groups,dc=example,dc=org' -s one -a find '(&(objectClass=groupOfNames))' uid member cn objectclass

```
Import realm json see
https://github.com/bitnami/charts/issues/5178

https://oauth2-proxy.github.io/oauth2-proxy/docs/configuration/oauth_provider#keycloak-auth-provider
https://github.com/bitnami/bitnami-docker-oauth2-proxy

## Keycloak
http://localhost:9100/

```
docker-compose -f docker-compose.keycloak.yml up
docker-compose up nginx voila-oauth2-proxy
docker-compose up nginx voila-oauth2-proxy label-studio
```

```
GET /? HTTP/1.1
Host: localhost:4180
Cache-Control: max-age=0
Cookie: _oauth2_proxy_0=sdsdsd; _oauth2_proxy_1=sdsdsdssd
Upgrade-Insecure-Requests: 1
X-Forwarded-Access-Token: ey....fN0_QO_S-aVn-az9T-Tkd_hg
X-Forwarded-Email: user01@localhost
X-Forwarded-For: 127.0.0.1
X-Forwarded-Groups: readers,offline_access,uma_authorization,default-roles-testrealm,role:readers,role:offline_access,role:uma_authorization,role:default-roles-testrealm,role:account:manage-account,role:account:manage-account-links,role:account:view-profile
X-Forwarded-Preferred-Username: user01
X-Forwarded-User: a4444444-1111-1221-a12a-123232323232


a6c71b85cad5
```

## NGINX
```
docker build -t nginx:latest docker-compose/nginx
```

```
docker-compose up 
```

## Label Studio
```
docker build -t label-studio:latest docker-compose/label-studio
```
http://localhost:9110/
 
## Voila
```
docker build -t voila:latest docker-compose/voila
docker run -p 9120:8888 -v `pwd`/docker-compose/voila/data:/data voila
```
Reach voila via oauth2-proxy:
http://localhost:4180/


## Monitoring

* [Open Grafana](http://localhost:3000)
* [Open Prometheus](http://localhost:9090)

![Dashboard](images/dashboard.png)
Inspired by [Jeremy Jordan
A simple solution for monitoring ML systems.
](https://www.jeremyjordan.me/ml-monitoring/)

## References:
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
