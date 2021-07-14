from bitnami/python:3.8

RUN pip install mlflow[extras]

EXPOSE 5000


# python -c "import boto3; cl = boto3.client('s3', endpoint_url='http://minio:9000', use_ssl=False);print(cl.list_buckets());"
# python -c "import boto3; cl = boto3.client('s3', aws_access_key_id='minio-access-key', aws_secret_access_key='minio-secret-key', endpoint_url='http://minio:9000', use_ssl=False);print(cl.list_buckets());"
# python -c "import os; print(os.environ['AWS_ACCESS_KEY_ID']);"
# export AWS_ACCESS_KEY_ID='minio-access-key'
# export AWS_SECRET_ACCESS_KEY='minio-secret-key'

CMD ["mlflow", "server", "--backend-store-uri", \
     "sqlite:////data/mydb.sqlite",  "--default-artifact-root", "s3://titanic/mlruns", \
     "--host", "0.0.0.0"]