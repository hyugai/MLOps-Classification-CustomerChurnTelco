This folder is used for local storage.
For "Prefect" local storage, in terminal type "conda env config vars set PREFECT_HOME=./cache/.prefect"
For "MLflow" local storage, in terminal when starting a local mlflow server, type "mlflow server --backend-store-uri sqlite:///cache/mlflow/mlflow.db", artifacts of runs for each experiment is set to "cache/mlflow"
