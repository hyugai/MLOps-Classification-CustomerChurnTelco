# data structures
import numpy as np
import pandas as pd

# model selection
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

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

# load base model
def load_basemodel() -> list:
    base_model = []
    base_model.append(('LR', LogisticRegression(n_jobs=-1)))
    base_model.append(('KNN', KNeighborsClassifier(n_jobs=-1)))
    base_model.append(('SVM', SVC()))
    base_model.append(('CART', DecisionTreeClassifier()))
    base_model.append(('ET', ExtraTreesClassifier(n_jobs=-1)))
    base_model.append(('RF', RandomForestClassifier(n_jobs=-1)))
    base_model.append(('GB', GradientBoostingClassifier()))
    base_model.append(('LGBM', LGBMClassifier(verbose=-1, n_jobs=-1)))
    base_model.append(('XGB', XGBClassifier(n_jobs=-1)))