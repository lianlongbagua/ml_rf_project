from src.targets_extraction import *
from src.features_extraction import *
from src.useful_tools import *

import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import pandas as pd
import numpy as np

def drop_ohlcv_cols(df: pd.DataFrame):
    """drop ohlcv columns"""
    return df.drop(
        columns=["open", "high", "low", "close", "volume", "open_interest"], axis=1
    )


def split_features_target(df: pd.DataFrame):
    """split features and target"""
    X = df.drop(['pos_change_signal', 'net_pos_signal', 'desired_pos_change', 'desired_pos_rolling'], axis=1)
    y = df[['pos_change_signal', 'net_pos_signal', 'desired_pos_change', 'desired_pos_rolling']]

    return X, y

if __name__ == "__main__":
    print(
        "This script will generate features from a csv file. \n \
        The csv file must contain the following columns: \n \
        datetime, open, high, low, close, volume, open_interest \n \
        The optimum positions are calculated using forward data, strictly speaking these are the best positions in hindsight \n \
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
        ")
    pth = input("Enter path to csv file, if using default named 'data.csv', press enter: ")
    if pth == "":
        pth = "data.csv"
    # read the 'data.csv' file in the current directory
    df = keep_essentials(pd.read_csv(pth))

    lags = input("Enter lags for feature engineering in the form of \
                 range(p, q, step): , default is range(10, 100, 10): ")
    if lags == "":
        lags = [i for i in range(10, 100, 10)]
    else:
        lags = list(eval(lags))

    preprocessing_pipeline = make_pipeline(
        FunctionTransformer(prepare_desired_pos, kw_args={"lag":50, "multiplier":10}),
        FunctionTransformer(generate_all_features_df, kw_args={"lags": lags}),
        FunctionTransformer(drop_ohlcv_cols),
        FunctionTransformer(split_features_target),
        verbose=True
    )

    feature_selector = SelectFromModel(ExtraTreesClassifier(n_estimators=1000, random_state=42, n_jobs=-1))

    training_classifier = RandomForestClassifier()
    training_regressor = RandomForestRegressor()

    print("Beginning preprocessing...")
    X, y = preprocessing_pipeline.fit_transform(df)
    print("Preprocessing complete")

    print("Beginning feature selection...")
    X = feature_selector.fit_transform(X, y['pos_change_signal'])
    print("Feature selection complete: dropped features: ", X.shape[1] - 1)

    print("leaving one set out for testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    print("Test set left out")

    ts_cv = TimeSeriesSplit(n_splits=5)
    print("Beginning training...")
    print("Training classifier for whether to enter into position...")
    grid_search_classifier = GridSearchCV(
        training_classifier,
        param_grid={
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [i for i in range(100, 5000, 500)],
            "min_samples_leaf": [i for i in range(100, 5000, 500)],
            "max_features": ["sqrt", "log2", "None"],
        },
        cv=ts_cv,
        scoring="accuracy",
        verbose=2,
        n_jobs=-1,
    )
    grid_search_classifier.fit(X_train, y_train['pos_change_signal'])
    print("Classifier for whether to enter into position trained")
    print('Mean score: ', grid_search_classifier.cv_results_['mean_test_score'])
    print("Best score: ", grid_search_classifier.best_score_)
    print("Best params: ", grid_search_classifier.best_params_)
    print("Best estimator: ", grid_search_classifier.best_estimator_)
    print("Best index: ", grid_search_classifier.best_index_)
    print("Saving model...")
    joblib.dump(grid_search_classifier.best_estimator_, 'models/enter_pos_classifier.pkl')

    print("Training classifier for whether to hold existing position...")
    grid_search_classifier.fit(X_train, y_train['net_pos_signal'])
    print("Classifier for whether to hold existing position trained")
    print('Mean score: ', grid_search_classifier.cv_results_['mean_test_score'])
    print("Best score: ", grid_search_classifier.best_score_)
    print("Best params: ", grid_search_classifier.best_params_)
    print("Best estimator: ", grid_search_classifier.best_estimator_)
    print("Best index: ", grid_search_classifier.best_index_)
    print("Saving model...")
    joblib.dump(grid_search_classifier.best_estimator_, 'models/hold_pos_classifier.pkl')

    print("Training regressor for how much to change position by...")
    grid_search_regressor = GridSearchCV(
        training_regressor,
        param_grid={
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [i for i in range(100, 5000, 500)],
            "min_samples_leaf": [i for i in range(100, 5000, 500)],
            "max_features": ["sqrt", "log2", "None"],
        },
        cv=ts_cv,
        scoring="R2",
        verbose=2,
        n_jobs=-1,
    )
    grid_search_regressor.fit(X_train, y_train['desired_pos_change'])
    print("Best score: ", grid_search_regressor.best_score_)
    print("Best params: ", grid_search_regressor.best_params_)
    print("Best estimator: ", grid_search_regressor.best_estimator_)
    print("Saving model...")
    joblib.dump(grid_search_regressor.best_estimator_, 'models/change_pos_regressor.pkl')

    print("Regressor for how much to change position by trained")
    grid_search_regressor.fit(X_train, y_train['desired_pos_rolling'])
    print("Best score: ", grid_search_regressor.best_score_)
    print("Best params: ", grid_search_regressor.best_params_)
    print("Best estimator: ", grid_search_regressor.best_estimator_)
    print("Saving model...")
    joblib.dump(grid_search_regressor.best_estimator_, 'models/hold_pos_regressor.pkl')

    print("Training complete")
    print("Summary of results:")
    print("Classifier for whether to enter into position:")
    print("Best estimator: ", grid_search_classifier.best_estimator_)
    print("Best score: ", grid_search_classifier.best_score_)
    print("Best params: ", grid_search_classifier.best_params_)
    print("Classifier for whether to hold existing position:")
    print("Best estimator: ", grid_search_classifier.best_estimator_)
    print("Best score: ", grid_search_classifier.best_score_)
    print("Best params: ", grid_search_classifier.best_params_)
    print("Regressor for how much to change position by:")
    print("Best estimator: ", grid_search_regressor.best_estimator_)
    print("Best score: ", grid_search_regressor.best_score_)
    print("Best params: ", grid_search_regressor.best_params_)
    print("Regressor for how much position to hold:")
    print("Best estimator: ", grid_search_regressor.best_estimator_)
    print("Best score: ", grid_search_regressor.best_score_)
    print("Best params: ", grid_search_regressor.best_params_)
    
    print("Beginning testing...")
    print("Testing classifier for whether to enter into position...")
    print("Score: ", grid_search_classifier.score(X_test, y_test['pos_change_signal']))
    print("Testing classifier for whether to hold existing position...")
    print("Score: ", grid_search_classifier.score(X_test, y_test['net_pos_signal']))
    print("Testing regressor for how much to change position by...")
    print("Score: ", grid_search_regressor.score(X_test, y_test['desired_pos_change']))
    print("Testing regressor for how much position to hold...")
    print("Score: ", grid_search_regressor.score(X_test, y_test['desired_pos_rolling']))
    print("Testing complete")
    
    print("Saving results...")
    results = pd.DataFrame({
        "Classifier for whether to enter into position": [
            grid_search_classifier.best_estimator_,
            grid_search_classifier.best_score_,
            grid_search_classifier.best_params_,
            grid_search_classifier.score(X_test, y_test['pos_change_signal'])
        ],
        "Classifier for whether to hold existing position": [
            grid_search_classifier.best_estimator_,
            grid_search_classifier.best_score_,
            grid_search_classifier.best_params_,
            grid_search_classifier.score(X_test, y_test['net_pos_signal'])
        ],
        "Regressor for how much to change position by": [
            grid_search_regressor.best_estimator_,
            grid_search_regressor.best_score_,
            grid_search_regressor.best_params_,
            grid_search_regressor.score(X_test, y_test['desired_pos_change'])
        ],
        "Regressor for how much position to hold": [
            grid_search_regressor.best_estimator_,
            grid_search_regressor.best_score_,
            grid_search_regressor.best_params_,
            grid_search_regressor.score(X_test, y_test['desired_pos_rolling'])
        ]
    })
    results.to_csv('results/results.csv')
    print("Results saved")

    print("Script complete")

