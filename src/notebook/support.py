# data structures
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# model selection
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

# metrics
from sklearn.metrics import fbeta_score, make_scorer, classification_report

# preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer

# impute
from sklearn.impute import SimpleImputer, KNNImputer

# compose
from sklearn.compose import ColumnTransformer

# resampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

# feature selection
from mlxtend.feature_selection import SequentialFeatureSelector

# pipeline
from imblearn.pipeline import Pipeline

# non-ensample algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# ensample algorithms
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier

# mlflow 
import mlflow
from mlflow.models import infer_signature, infer_pip_requirements

# others
import pickle, uuid
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Callable

# UDF: prepare data to train model
def prepare_data_to_train(path: str) -> dict:
    materials = dict()
    arr = pd.read_csv(path).values
    ##
    le = LabelEncoder()
    X, y = arr[:, :-1], le.fit_transform(arr[:, -1])
    ##
    materials['X_train'], materials['X_test'], materials['y_train'], materials['y_test'] = train_test_split(X, y, test_size=0.3, stratify=y, random_state=7)
    
    return materials

# UDF: load base model
def load_base_models() -> list:
    base_models = []
    base_models.append(('LR', LogisticRegression(n_jobs=-1)))
    base_models.append(('KNN', KNeighborsClassifier(n_jobs=-1)))
    base_models.append(('SVM', SVC()))
    base_models.append(('CART', DecisionTreeClassifier()))
    base_models.append(('ET', ExtraTreesClassifier(n_jobs=-1)))
    base_models.append(('RF', RandomForestClassifier(n_jobs=-1)))
    base_models.append(('GB', GradientBoostingClassifier()))
    base_models.append(('LGBM', LGBMClassifier(verbose=-1, n_jobs=-1)))
    base_models.append(('XGB', XGBClassifier(n_jobs=-1)))

    return base_models

# UDF: get selected models
def get_seleted_models(names: list) -> list:
    base_models = load_base_models()
    selected_base_models = []
    ##
    for name, model in base_models:
        if name in names:
            selected_base_models.append((name, model))

    return selected_base_models

# UDF: get kfold results
def get_kfold_results(models: list, X: np.ndarray, y: np.ndarray) -> dict:
    kfold_results = dict()
    ##
    for name, model in models:
        cv_results = cross_val_score(model, X, y, 
                                     cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3), 
                                     scoring=make_scorer(fbeta_score, beta=2))
        kfold_results[name] = cv_results
        print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

    return kfold_results

# UDF: plot kfold results
def plot_kfold_results(kfold_results: dict, title: str=None):
    data_to_plot = pd.DataFrame(kfold_results)
    ##
    sorted_idxes = data_to_plot.mean()\
        .sort_values(ascending=False).index.tolist()
    data_to_plot = data_to_plot.reindex(sorted_idxes, axis=1)
    ##
    g = sns.boxplot(data_to_plot)
    g.set_title(title, fontdict=dict(size=15))

    plt.show()

# UDF: dump model
def dump_model(model, path: str) -> None:
    with open(path, 'wb') as output:
        pickle.dump(obj=model, file=output)

# class: Node
"""
Each node will represent as the validation test for a specific type transformer to select all or just a subset of the input features.
To invoke the validation test, each node will need 4 paramaters inlcluding 'name', 'model', 'test' and 'data'. The last one can be passed later in another class.
When the test finished, we'll have 2 output including passed features and failed ones (as indexs). In case of complex sequential nodes, the failed ones will be passed to the next node repeatedly.
Only pass failed features' indexes when the next is available.
Only return empty list for the passed one when there is no feature getting through the test.
"""
class Node:
    def __init__(self, name, model) -> None:
        self.name, self.model = name, model
        self.data, self.next = dict(idxes=[]), None
        self.test: Callable[[dict], tuple[list, list]]
    ##
    def validate(self):
        passed, failed = self.test(self.data)
        ### check if we should pass the failed ones to the next node.
        if self.next != None:
            self.next.data.update({'idxes': failed, 
                                'X': self.data['X']})
        ### check if any feature passed the test to decide if return the empty list or not. 
        if len(passed) != 0:
            return [(self.name, self.model, passed)]
        else:
            return []
        
# class: InitBinaryTree
"""
This tree play the vital role as the glue to connect all the separate or partially separate nodes into 2 branch which is the numeric branch and categorial one.
When being initialized, it will take a list of node for mumeric branch, similar for the categorical one.
It also play the role as third party to pass data to the first node of each branch by approach those nodes (as properties) through this class.
Each parent branch will be invoked separately to prevent nodes from being overriden.
"""
class InitBinaryTree():
    def __init__(self, left: list, right: list) -> None:
        self.left = left
        self.right = right
    ## we will take a specific info from data of each node as signal to decide if we should invoke the node, that is the number of indexes existing in each node waiting to be validated. If they present, the tree will capture the singnal and invoke the node, otherwise the execution will be terminated.
    def invoke_validation_of_branch(self, nodes: list, passed: dict) -> dict:
        num_iters, i = len(nodes), 0
        signal = len(nodes[i].data['idxes'])

        while signal != 0:
            passed[nodes[i].name] = nodes[i].validate()

            i += 1
            ### the bug is here
            if i < num_iters: 
                signal = len(nodes[i].data['idxes'])
            else:
                break

        return passed
    ##
    def validate(self):
        passed = dict()
        self.invoke_validation_of_branch(self.left, passed)
        self.invoke_validation_of_branch(self.right, passed)
        ### filter out the empty results, make it eligible to become the input of class ColumnTrasnformer
        results = [result[0] for result in passed.values() if len(result) != 0]

        return results

# UDC: base customized transformer for sequatial feature selection task
class FSBaseTransformer(BaseEstimator, TransformerMixin):
    ##
    def __init__(self,tree: InitBinaryTree) -> None:
        self.tree = tree

        return None
    ##
    def check_ndim(self, X: np.ndarray) -> tuple[np.ndarray, int | float]:
        ndim = X.ndim
        if ndim == 2:
            return X, X.shape[1]
        else:
            return X.reshape(-1, 1), 1
    ##
    def detect_category(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        X, num_iters = self.check_ndim(X)
        ###
        idxes = dict(num=[], cat=[])
        ### 
        for i in range(num_iters):
            try:
                float(X[0, i])
                idxes['num'].append(i)
            except:
                idxes['cat'].append(i)

        return X, idxes
    ##
    def fit(self, X: np.ndarray, y=None):
        self.ct = ColumnTransformer(self.assigned_transformers, remainder='passthrough')\
            .fit(X)
        
        return self
    ##
    def transform(self, X: np.ndarray, y=None):
        X, _ = self.check_ndim(X)

        return self.ct.transform(X)

#  class: SFS
"""
"""
class SFS(FSBaseTransformer):
    def fit(self, X: np.ndarray, y=None):
        X, idxes = self.detect_category(X)

        self.tree.left[0].data.update({'X': X, 'idxes': idxes['num']})
        self.tree.right[0].data.update({'X': X, 'idxes': idxes['cat']})
        self.assigned_transformers = self.tree.validate()

        super().fit(X)

        return self