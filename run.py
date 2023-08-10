from src.targets_extraction import *
from src.features_extraction import *
from src.useful_tools import *

import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from evolutionary_search import EvolutionaryAlgorithmSearchCV
from skopt import BayesSearchCV

import pandas as pd


if __name__ == "__main__":
    print(
        "This script will generate features from a csv file. \n \
        The csv file must contain the following columns: \n \
        datetime, open, high, low, close, volume, open_interest \n \
        The optimum positions are calculated using forward data, strictly \
        speaking these are the best positions in hindsight \n \
        The script will then select the most relevant features using ExtraTreesClassifier \n \
        It will then begin training 4 models on the data. \n \
        The models are: \n \
        \t 1. Random Forest Classifier for whether to enter into position\n \
        \t 2. Random Forest Classifier for whether to hold existing position\n \
        \t 3. Random Forest Regressor for how much to change position by\n \
        \t 4. Random Forest Regressor for how much position to hold\n \
        for the classification models, both a prediction and a degree of confidence will be given\n \
        for the regression models, only a prediction will be given\n \
        The models will be saved to the models folder\n \
        cross-validation will be performed on the models\n \
        The results of the cross-validation will be saved to the results folder\n \
        "
    )
    pth = input(
        "Enter path to csv file, if using default named 'data.csv', press enter: "
    )
    if pth == "":
        pth = "data.csv"
    # read the 'data.csv' file in the current directory
    df = keep_essentials(pd.read_csv(pth))

    train_test_ratio = input(
        "Enter how much data you want to reserve for testing, default is 0.15:"
    )

    lags = input(
        "Enter lags for feature engineering in the form of \
        range(p, q, step): , default is range(10, 500, 10): "
    )
    if lags == "":
        lags = [i for i in range(10, 500, 10)]
    else:
        lags = list(eval(lags))

    cv_mode = input(
        "Enter cross-validation mode, default is 'BayesSearchCV': "
    )
    if cv_mode == "":
        cv_mode = "BayesSearchCV"

    preprocessing_pipeline = make_pipeline(
        FunctionTransformer(prepare_desired_pos, kw_args={"lag": 50, "multiplier": 10}),
        FunctionTransformer(generate_all_features_df, kw_args={"lags": lags}),
        FunctionTransformer(drop_ohlcv_cols),
        FunctionTransformer(split_features_target),
        verbose=True,
    )

    feature_selector = SelectFromModel(
        ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    )

    training_classifier = RandomForestClassifier()
    training_regressor = RandomForestRegressor()

    print("Beginning preprocessing...")
    X, y = preprocessing_pipeline.fit_transform(df)
    print("Preprocessing complete")

    print("Beginning feature selection...")
    X = feature_selector.fit_transform(X, y["pos_change_signal"])
    print("Feature selection complete: dropped features: ", X.shape[1] - 1)
    print("features selected: ", feature_selector.feature_names_in_)
    print("Saving feature names selected to feature_names.txt...")
    with open("feature_names.txt", "w") as f:
        f.write(str(feature_selector.feature_names_in_))


    print("leaving one set out for testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(train_test_ratio), shuffle=False
    )
    print("Test set left out")

    ts_cv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [i for i in range(100, 5000, 500)],
        "min_samples_leaf": [i for i in range(100, 5000, 500)],
        "max_features": ["sqrt", "log2", None],
    }

    print("Beginning training...")
    print("Training classifier for whether to enter into position...")
    if cv_mode == "GridSearchCV":
        classifier_optimizer = GridSearchCV(
            training_classifier,
            param_grid=param_grid,
            cv=ts_cv,
            scoring="accuracy",
            verbose=2,
            n_jobs=-1,
        )
    elif cv_mode == "EvolutionaryAlgorithmSearchCV":
        classifier_optimizer = EvolutionaryAlgorithmSearchCV(
            training_classifier,
            params=param_grid,
            cv=ts_cv,
            scoring="accuracy",
            verbose=2,
            n_jobs=-1,
            population_size=50,
            gene_mutation_prob=0.10,
            gene_crossover_prob=0.5,
            tournament_size=3,
            generations_number=30,
        )
    elif cv_mode == "BayesSearchCV":
        classifier_optimizer = BayesSearchCV(
            training_classifier,
            param_grid,
            cv=ts_cv,
            scoring="accuracy",
            verbose=2,
            n_jobs=-1,
            n_iter=50,
        )
    classifier_optimizer.fit(X_train, y_train["pos_change_signal"])
    print("Classifier for whether to enter into position trained")
    print("Mean score: ", classifier_optimizer.cv_results_["mean_test_score"])
    print("Best score: ", classifier_optimizer.best_score_)
    print("Best params: ", str((classifier_optimizer.best_params_)))
    print("Saving model...")
    joblib.dump("models/enter_pos_classifier.pkl")

    print("Training classifier for whether to hold existing position...")
    classifier_optimizer.fit(X_train, y_train["net_pos_signal"])
    print("Classifier for whether to hold existing position trained")
    print("Mean score: ", classifier_optimizer.cv_results_["mean_test_score"])
    print("Best score: ", classifier_optimizer.best_score_)
    print("Best params: ", str((classifier_optimizer.best_params_)))
    print("Saving model...")
    joblib.dump("models/hold_pos_classifier.pkl")

    print("Training regressor for how much to change position by...")
    if cv_mode == "GridSearchCV":
        regressor_optimzier = GridSearchCV(
            training_regressor,
            params=param_grid,
            cv=ts_cv,
            scoring="R2",
            verbose=2,
            n_jobs=-1,
        )
    elif cv_mode == "EvolutionaryAlgorithmSearchCV":
        regressor_optimzier = EvolutionaryAlgorithmSearchCV(
            training_regressor,
            param_grid=param_grid,
            cv=ts_cv,
            scoring="R2",
            verbose=True,
            n_jobs=-1,
            population_size=50,
            gene_mutation_prob=0.10,
            gene_crossover_prob=0.5,
            tournament_size=3,
            generations_number=30,
        )
    elif cv_mode == "BayesSearchCV":
        regressor_optimzier = BayesSearchCV(
            training_regressor,
            param_grid,
            cv=ts_cv,
            scoring="R2",
            verbose=2,
            n_jobs=-1,
            n_iter=50,
        )
    regressor_optimzier.fit(X_train, y_train["desired_pos_change"])
    print("Best score: ", regressor_optimzier.best_score_)
    print("Best params: ", str(regressor_optimzier.best_params_))
    print("Saving model...")
    joblib.dump("models/change_pos_regressor.pkl")

    print("Regressor for how much to change position by trained")
    regressor_optimzier.fit(X_train, y_train["desired_pos_rolling"])
    print("Best score: ", regressor_optimzier.best_score_)
    print("Best params: ", str(regressor_optimzier.best_params_))
    print("Saving model...")
    joblib.dump("models/hold_pos_regressor.pkl")

    print("Training complete")
    print("Summary of results:")
    print("Classifier for whether to enter into position:")
    print("Best estimator: ", classifier_optimizer.best_estimator_)
    print("Best score: ", classifier_optimizer.best_score_)
    print("Best params: ", str(classifier_optimizer.best_params_))
    print("Classifier for whether to hold existing position:")
    print("Best estimator: ", classifier_optimizer.best_estimator_)
    print("Best score: ", classifier_optimizer.best_score_)
    print("Best params: ", str(classifier_optimizer.best_params_))
    print("Regressor for how much to change position by:")
    print("Best estimator: ", regressor_optimzier.best_estimator_)
    print("Best score: ", regressor_optimzier.best_score_)
    print("Best params: ", str(regressor_optimzier.best_params_))
    print("Regressor for how much position to hold:")
    print("Best estimator: ", regressor_optimzier.best_estimator_)
    print("Best score: ", regressor_optimzier.best_score_)
    print("Best params: ", str(regressor_optimzier.best_params_))

    print("Beginning testing...")
    print("Testing classifier for whether to enter into position...")
    print("Score: ", classifier_optimizer.score(X_test, y_test["pos_change_signal"]))
    print("Testing classifier for whether to hold existing position...")
    print("Score: ", classifier_optimizer.score(X_test, y_test["net_pos_signal"]))
    print("Testing regressor for how much to change position by...")
    print("Score: ", regressor_optimzier.score(X_test, y_test["desired_pos_change"]))
    print("Testing regressor for how much position to hold...")
    print("Score: ", regressor_optimzier.score(X_test, y_test["desired_pos_rolling"]))
    print("Testing complete")

    print("Saving results...")
    results = pd.DataFrame(
        {
            "Classifier for whether to enter into position": [
                classifier_optimizer.best_score_,
                str(classifier_optimizer.best_params_),
                classifier_optimizer.score(X_test, y_test["pos_change_signal"]),
            ],
            "Classifier for whether to hold existing position": [
                classifier_optimizer.best_score_,
                str(classifier_optimizer.best_params_),
                classifier_optimizer.score(X_test, y_test["net_pos_signal"]),
            ],
            "Regressor for how much to change position by": [
                regressor_optimzier.best_score_,
                str(regressor_optimzier.best_params_),
                regressor_optimzier.score(X_test, y_test["desired_pos_change"]),
            ],
            "Regressor for how much position to hold": [
                regressor_optimzier.best_score_,
                str(regressor_optimzier.best_params_),
                regressor_optimzier.score(X_test, y_test["desired_pos_rolling"]),
            ],
        }
    )
    results.to_csv("results/results.csv")
    print("Results saved")

    print("Script complete")
