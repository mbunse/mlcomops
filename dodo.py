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
            
        ]
    }
