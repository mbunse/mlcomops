from bitnami/python:3.8

RUN pip install mlflow[extras]

RUN chgrp -R 0 . && \
    chmod -R g+rwX .

USER 1001

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", \
     "sqlite:////data/mydb.sqlite",  "--default-artifact-root", "s3://titanic/mlruns", \
     "--host", "0.0.0.0"]