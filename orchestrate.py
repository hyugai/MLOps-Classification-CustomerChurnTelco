# UDF and UDC
from src.notebook.support import *
from src.mlflow.support import *
from src.prefect.data_wrangling import *
from src.prefect.model_engineering import *
# prefect
from prefect.flows import flow
from prefect.tasks import task

# data wrangling
def load_raw_data(path: str) -> dict[pd.DataFrame]:
    materials = {'df': pd.read_csv(path)}

    return materials

def quality_testing(materials: dict) -> dict:
    return materials