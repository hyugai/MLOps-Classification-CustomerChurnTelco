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

# imputation
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer

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

# others
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# UDF: prepare data to train model
def prepare_data_to_train(path: str) -> dict:
    materials = dict()
    arr = pd.read_csv(path).values
    ##
    le = LabelEncoder()
    X, y = arr[:, :-1], le.fit_transform(arr[:, -1])
    ##
    materials['X_train'], materials['X_test'], materials['y_train'], materials['y_test'] = train_test_split(X, y, 
                                                                                                            test_size=0.3, stratify=y, random_state=7)
    
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

    return base_models

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

# UDC: base customized transformer for sequatial feature selection task
class SFSBaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformers: dict) -> None:
        materials = {
            'transformers': transformers
        }
        self.materials = materials

    def _check_ndim(self, X: np.ndarray):
        ndim = X.ndim
        if ndim == 2:
            return X, X.shape[1]
        else:
            return X.reshape(-1, 1), 1

    def _detect_category(self, X: np.ndarray):
        X, num_iters = self._check_ndim(X)
        ##
        self.materials['num_idxes'], self.materials['cat_idxes'] = [], []
        ## 
        for i in range(num_iters):
            try:
                X[0, i].astype(float)
                self.materials['num_idxes'].append(i)
            except:
                self.materials['cat_idxes'].append(i)
