from bitnami/python:3.8

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
RUN pip install dvc[s3] scikit-learn pandas fairlearn cloudpickle \
    fastapi uvicorn[standard] prometheus-fastapi-instrumentator lime

COPY .dvc/config .dvc/config
COPY dvc.yaml .
COPY dvc.lock .
COPY app.py .

RUN dvc config core.no_scm true && \
    dvc remote modify --local minio access_key_id $AWS_ACCESS_KEY_ID && \
    dvc remote modify --local minio secret_access_key $AWS_SECRET_ACCESS_KEY && \
    dvc pull models/model.pkl models/explainer.pkl models/outlier_detector.pkl 

RUN chgrp -R 0 . && \
    chmod -R g+rwX .

USER 1001

EXPOSE 8080

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "app:app"]