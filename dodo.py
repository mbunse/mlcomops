def task_setup_dvc():
    """Setup dvc"""

    def create_bucket():
        from minio import Minio
        client = Minio(
            "localhost:9000",
            secure=False,
            access_key="minio-access-key",
            secret_key="minio-secret-key",
        )
        if client.bucket_exists("titanic"):
            client.remove_bucket("titanic")
        client.make_bucket("titanic")

    return {
        'actions': [
            create_bucket,
            "dvc init -f", 
            "dvc remote add -d minio s3://titanic/dvcrepo",
            "dvc remote modify minio endpointurl http://localhost:9000",
            "dvc remote modify --local minio access_key_id minio-access-key",
            "dvc remote modify --local minio secret_access_key minio-secret-key",
            ],
        'targets': [".dvc/config"],
        }

def task_run_steps():
    return {
        "actions": [
            "dvc run -n load_data --force -o ../data/interim/train_df.pkl -o ../data/interim/valid_df.pkl " + \
                "-o ../data/interim/outlier_df.pkl -d load_data.pct.py -w notebooks python load_data.pct.py",
            "dvc run -n train --force -d ../data/interim/train_df.pkl -d train.pct.py -M ../models/score.json " + \
                "-o ../models/model.pkl -o ../models/feat_names.json -w notebooks python train.pct.py",
            "dvc run -n prepare_explainer --force -d ../models/model.pkl -d ../data/interim/train_df.pkl " + \
                "-d ../data/interim/valid_df.pkl -d ../models/feat_names.json -o ../models/explainer.pkl " + \
                "-w notebooks python prepare_explainer.pct.py",
            "dvc run -n outlier_model --force -w notebooks -d ../data/interim/train_df.pkl " +  \
                "-d ../data/interim/valid_df.pkl -d ../models/feat_names.json -d ../models/model.pkl " + \
                "-d ../data/interim/outlier_df.pkl -o ../models/outlier_detector.pkl python outlier_detector.pct.py",
        ]
    }

def task_build_modelapi():
    return {
        "actions": [
            "docker build --network=host --build-arg AWS_ACCESS_KEY_ID=minio-access-key --build-arg AWS_SECRET_ACCESS_KEY=minio-secret-key -t modelapi:latest ."
        ]
    }

def task_generate_api():
    return {
        "actions":  [
            "openapi-python-client generate --url http://127.0.0.1:8080/openapi.json"
        ]
    }