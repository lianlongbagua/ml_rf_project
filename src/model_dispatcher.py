from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV
)
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Real, Categorical
from skopt import BayesSearchCV


models = {
    "DecisionTreeClassifier": tree.DecisionTreeClassifier(),
    "RandomForestClassifier": ensemble.RandomForestClassifier(),
    "ExtraTreesClassifier": ensemble.ExtraTreesClassifier(),
    "AdaBoostClassifier": ensemble.AdaBoostClassifier(),
    "GradientBoostingClassifier": ensemble.GradientBoostingClassifier(),
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

