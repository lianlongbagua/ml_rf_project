from targets_extraction import *
from features_extraction import *
from useful_tools import *
import config
import argparse

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd


if __name__ == "__main__":
    print("Script started")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="RandomForest",
        help="type of model to use for training",
    )

    parser.add_argument(
        "--cv",
        type=str,
        default="GridSearchCV",
        help="type of cross validation to use for training",
    )

    parser.add_argument(
        "--lag",
        type=int,
        default=10,
        help="lag for desired position (how many periods to look into the future)",
    )

    args = parser.parse_args()
    training_classifier, training_regressor = config.model_combos[args.model]

    df = keep_essentials(pd.read_csv(config.TRAINING_DATA))

    lags = config.LAGS
    
    lag_for_desired_pos = args.lag
    multiplier_for_desired_pos = 10

    preprocessing_pipeline = make_pipeline(
        FunctionTransformer(prepare_desired_pos, kw_args={"lag": lag_for_desired_pos, "multiplier": multiplier_for_desired_pos}),
        FunctionTransformer(generate_all_features_df, kw_args={"lags": lags}),
        FunctionTransformer(drop_ohlcv_cols),
        FunctionTransformer(split_features_target),
        verbose=True,
    )

    feature_selector = SelectFromModel(
        ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    )

    print("Beginning preprocessing...")
    X, y = preprocessing_pipeline.fit_transform(df)
    print("Preprocessing complete")

    print("Beginning feature selection...")
    preselected_feats = X.shape[1]
    X = feature_selector.fit_transform(X, y["pos_change_signal"])
    print("feature selection complete. number of dropped features", X.shape[1] - preselected_feats)
    print("features selected: ", feature_selector.feature_names_in_)
    print("Saving feature names selected to feature_names.txt...")
    with open("feature_names.txt", "w") as f:
        i = 0
        for string in feature_selector.feature_names_in_:
            f.write(str(i) + "_" + string + ",")
            i += 1

    ts_cv = TimeSeriesSplit(n_splits=5)

    params = config.PARAM_GRID_TREE

    print("Beginning training...")
    print("Training classifier for whether to enter into position...")
    classifier_optimizer = config.cross_validation[args.cv](
        training_classifier,
        params,
        cv=ts_cv,
        scoring="accuracy",
        verbose=2,
        n_jobs=-1,
    )

    classifier_optimizer.fit(X, y["pos_change_signal"])
    print("Classifier for whether to enter into position trained")
    print("Mean score: ", classifier_optimizer.cv_results_["mean_test_score"])
    print("Best score: ", classifier_optimizer.best_score_)
    print("Best params: ", str((classifier_optimizer.best_params_)))
    print("Saving training specs...")
    pd.DataFrame(classifier_optimizer.cv_results_).to_csv("../model/cls_enterpos_training_specs.csv")

    print("Training classifier for whether to hold existing position...")
    classifier_optimizer.fit(X, y["net_pos_signal"])
    print("Classifier for whether to hold existing position trained")
    print("Mean score: ", classifier_optimizer.cv_results_["mean_test_score"])
    print("Best score: ", classifier_optimizer.best_score_)
    print("Best params: ", str((classifier_optimizer.best_params_)))
    pd.DataFrame(classifier_optimizer.cv_results_).to_csv("../model/cls_holdpos_training_specs.csv")

    regressor_optimizer = config[args.cv](
        estimator=training_regressor,
        param_grid=params,
        cv=ts_cv,
        scoring="R2",
        verbose=2,
        n_jobs=-1,
    )
    print("Training regressor for how much to change position")
    regressor_optimizer.fit(X, y["desired_pos_change"])
    print("Best score: ", regressor_optimizer.best_score_)
    print("Best params: ", str(regressor_optimizer.best_params_))
    pd.DataFrame(regressor_optimizer.cv_results_).to_csv("../model/cls_chpos_training_specs.csv")
    print("Parameters saved")

    print("Training regressor for how much to hold position")
    regressor_optimizer.fit(X, y["desired_pos_rolling"])
    print("Best score: ", regressor_optimizer.best_score_)
    print("Best params: ", str(regressor_optimizer.best_params_))
    pd.DataFrame(regressor_optimizer.cv_results_).to_csv("../model/cls_chhold_training_specs.csv")
    print("Parameters saved")

    print("Training complete")
    print("Summary of results:")
    print("Classifier for whether to enter into position:")
    print("Best score: ", classifier_optimizer.best_score_)
    print("Best params: ", str(classifier_optimizer.best_params_))
    print("Classifier for whether to hold existing position:")
    print("Best score: ", classifier_optimizer.best_score_)
    print("Best params: ", str(classifier_optimizer.best_params_))
    print("Regressor for how much to change position by:")
    print("Best score: ", regressor_optimizer.best_score_)
    print("Best params: ", str(regressor_optimizer.best_params_))
    print("Regressor for how much position to hold:")
    print("Best score: ", regressor_optimizer.best_score_)
    print("Best params: ", str(regressor_optimizer.best_params_))

    print("Saving results...")
    results = pd.DataFrame(
        {
            "Classifier for whether to enter into position": [
                classifier_optimizer.best_score_,
                str(classifier_optimizer.best_params_),
            ],
            "Classifier for whether to hold existing position": [
                classifier_optimizer.best_score_,
                str(classifier_optimizer.best_params_),
            ],
            "Regressor for how much to change position by": [
                regressor_optimizer.best_score_,
                str(regressor_optimizer.best_params_),
            ],
            "Regressor for how much position to hold": [
                regressor_optimizer.best_score_,
                str(regressor_optimizer.best_params_),
            ],
        }
    )
    results.to_csv("../model/results.csv")
    print("Results saved")

    print("Script complete")
