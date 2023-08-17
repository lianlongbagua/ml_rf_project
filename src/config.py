from sklearn import tree
from sklearn import ensemble
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV
)
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from skopt import BayesSearchCV

"""
Some boilerplate configs for the project
"""

TRAINING_DATA = "../data/data.csv"

LAGS = [10, 20]

PARAM_GRID_TREE = {
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [i for i in range(100, 2000, 100)],
    "min_samples_leaf": [i for i in range(100, 2000, 100)],
    "max_features": ["sqrt", "log2", None],
}

GENETIC_PARAM_GRID_TREE = {
"max_depth": Integer(5, 6),
"min_samples_split": Integer(100, 200),
"max_features": Categorical(["log2"]),
}

"""
Models and CV's to use
"""

model_combos = {
    "DecisionTree": (tree.DecisionTreeClassifier(), tree.DecisionTreeRegressor()),
    "RandomForest": (ensemble.RandomForestClassifier(), ensemble.RandomForestRegressor()),
    "ExtraTrees": (ensemble.ExtraTreesClassifier(), ensemble.ExtraTreesRegressor()),
    "AdaBoost": (ensemble.AdaBoostClassifier(), ensemble.AdaBoostRegressor()),
    "GradientBoosting": (ensemble.GradientBoostingClassifier(), ensemble.GradientBoostingRegressor()),
}

models = {
    "DecisionTreeClassifier": tree.DecisionTreeClassifier(),
    "RandomForestClassifier": ensemble.RandomForestClassifier(),
    "ExtraTreesClassifier": ensemble.ExtraTreesClassifier(),
    "AdaBoostClassifier": ensemble.AdaBoostClassifier(),
    "GradientBoostingClassifier": ensemble.GradientBoostingClassifier(),
    "DecisionTreeRegressor": tree.DecisionTreeRegressor(),
    "RandomForestRegressor": ensemble.RandomForestRegressor(),
    "ExtraTreesRegressor": ensemble.ExtraTreesRegressor(),
    "AdaBoostRegressor": ensemble.AdaBoostRegressor(),
    "GradientBoostingRegressor": ensemble.GradientBoostingRegressor(),
}

cross_validation = {
    "GridSearchCV": GridSearchCV,
    "RandomizedSearchCV": RandomizedSearchCV,
    "HalvingGridSearchCV": HalvingGridSearchCV,
    "HalvingRandomSearchCV": HalvingRandomSearchCV,
    "BayesSearchCV": BayesSearchCV,
    "GASearchCV": GASearchCV,
}