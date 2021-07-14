from bitnami/python:3.8

RUN pip install dvc[s3] transformers[torch] fastapi uvicorn[standard] prometheus-fastapi-instrumentator

COPY .dvc/config .dvc/config
COPY models/model.dvc models/model.dvc
COPY app.py .

RUN dvc config core.no_scm true && \
    dvc pull models/model.dvc

EXPOSE 8080

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "app:app"]