# UDF and UDC
## add directory to src
import os, sys
cwd = os.getcwd()
os.chdir('../..')
path_to_src = os.getcwd()
if path_to_src not in sys.path:
    sys.path.append(path_to_src)
os.chdir(cwd)
## udf and udc
from src.notebook.support import *
from src.mlflow.support import *
from src.prefect.data_wrangling import *
from src.prefect.model_engineering import *
# others
from functools import wraps
from typing import Callable

# udf: format adjustment
def adjust_format(func: Callable[[str], dict]) -> dict:
    @wraps(func)
    def wrapper(*args, **kargs):
        materials = func(*args, **kargs)

        