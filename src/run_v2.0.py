from targets_extraction import *
from features_extraction import *
from useful_tools import *
from preprocessing import prep_data
import config
import argparse

from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
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
        help="hyperparameter optimization methods",
    )

    args = parser.parse_args()

    training_classifier, training_regressor = config.model_combos[args.model]

    df = pd.read_csv(config.TRAINING_DATA)

    ts_cv = StratifiedKFold(n_splits=5)

    print("Beginning preprocessing...")
    X, y = prep_data(df, lags, future_pred, multiplier_for_desired_pos)
    print("Preprocessing complete")

    feature_selector = RFECV(
        estimator=ExtraTreesClassifier(n_estimators=100),
        step=5,
        cv=5,
        n_jobs=-1,
        verbose=2,
    )

    print("Beginning feature selection...")
    preselected_feats = X.shape[1]
    feature_selector.fit(X, y["pos_change_signal"])
    print(
        "feature selection complete. number of dropped features",
        preselected_feats - X.shape[1]
    )
    print("features selected: ", feature_selector.get_feature_names_out())
    print("Saving feature names selected to feature_names.txt...")
    with open("feature_names.txt", "w") as f:
        for i, string in enumerate(feature_selector.get_feature_names_out()):
            f.write(str(i - 1) + "_" + string + ",")
    print("Feature names saved")
    X = feature_selector.transform(X)

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
    pd.DataFrame(classifier_optimizer.cv_results_
                 ).to_csv("../model/cls_enterpos_training_specs.csv")

    print("Training classifier for whether to hold existing position...")
    classifier_optimizer.fit(X, y["net_pos_signal"])
    print("Classifier for whether to hold existing position trained")
    print("Mean score: ", classifier_optimizer.cv_results_["mean_test_score"])
    print("Best score: ", classifier_optimizer.best_score_)
    print("Best params: ", str((classifier_optimizer.best_params_)))
    pd.DataFrame(classifier_optimizer.cv_results_
                 ).to_csv("../model/cls_holdpos_training_specs.csv")

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
    pd.DataFrame(regressor_optimizer.cv_results_
                 ).to_csv("../model/cls_chpos_training_specs.csv")
    print("Parameters saved")

    print("Training regressor for how much to hold position")
    regressor_optimizer.fit(X, y["desired_pos_rolling"])
    print("Best score: ", regressor_optimizer.best_score_)
    print("Best params: ", str(regressor_optimizer.best_params_))
    pd.DataFrame(regressor_optimizer.cv_results_
                 ).to_csv("../model/cls_chhold_training_specs.csv")
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
    results = pd.DataFrame({
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
    })
    results.to_csv("../model/results.csv")
    print("Results saved")

    print("Script complete")
