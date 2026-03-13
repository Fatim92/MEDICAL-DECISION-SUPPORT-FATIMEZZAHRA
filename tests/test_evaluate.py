# tests/test_evaluate.py

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import optuna 
import matplotlib.pyplot as plt
from unittest.mock import patch
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from src.evaluate import cv_score_model_with_earlystop
from src.evaluate import objective_lgb
from src.evaluate import objective_xgb
from src.evaluate import run_optuna_tuning
from src.evaluate import plot_and_save_confusion
from src.evaluate import plot_and_save_roc
from src.evaluate import fit_final_and_evaluate, main
##✅✅

# ── fixtures ────────────────────────────────────────────────────────────────
## Test of cv_score_model_with_earlystop function 
@pytest.fixture
def dummy_data():
    """Small balanced 7-class dataset that mirrors the obesity dataset."""
    import pandas as pd
    X, y = make_classification(
        n_samples=210,
        n_features=10,
        n_classes=7,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    return X, y


# ── return shape / type tests ────────────────────────────────────────────────

def test_returns_three_values(dummy_data):
    X, y = dummy_data
    result = cv_score_model_with_earlystop(LogisticRegression, {}, X, y, n_splits=3)
    assert len(result) == 3, "Should return (mean_f1, oof_preds, oof_proba)"


def test_mean_f1_is_float(dummy_data):
    X, y = dummy_data
    mean_f1, _, _ = cv_score_model_with_earlystop(LogisticRegression, {}, X, y, n_splits=3)
    assert isinstance(mean_f1, float)


def test_f1_between_0_and_1(dummy_data):
    X, y = dummy_data
    mean_f1, _, _ = cv_score_model_with_earlystop(LogisticRegression, {}, X, y, n_splits=3)
    assert 0.0 <= mean_f1 <= 1.0


def test_oof_preds_shape(dummy_data):
    X, y = dummy_data
    _, oof_preds, _ = cv_score_model_with_earlystop(LogisticRegression, {}, X, y, n_splits=3)
    assert oof_preds.shape == (len(y),), "oof_preds must cover every sample"


def test_oof_proba_shape(dummy_data):
    X, y = dummy_data
    _, _, oof_proba = cv_score_model_with_earlystop(LogisticRegression, {}, X, y, n_splits=3)
    assert oof_proba is not None
    assert oof_proba.shape == (len(y), 7), "oof_proba shape should be (n_samples, n_classes)"


def test_oof_preds_are_integers(dummy_data):
    X, y = dummy_data
    _, oof_preds, _ = cv_score_model_with_earlystop(LogisticRegression, {}, X, y, n_splits=3)
    assert oof_preds.dtype == int


# ── model-specific tests ─────────────────────────────────────────────────────

def test_works_with_xgboost(dummy_data):
    X, y = dummy_data
    params = {"n_estimators": 50, "verbosity": 0, "use_label_encoder": False}
    mean_f1, oof_preds, oof_proba = cv_score_model_with_earlystop(
        XGBClassifier, params, X, y, n_splits=3
    )
    assert 0.0 <= mean_f1 <= 1.0
    assert oof_preds.shape == (len(y),)


def test_works_with_lightgbm(dummy_data):
    X, y = dummy_data
    params = {"n_estimators": 50, "verbose": -1}
    mean_f1, oof_preds, oof_proba = cv_score_model_with_earlystop(
        lgb.LGBMClassifier, params, X, y, n_splits=3
    )
    assert 0.0 <= mean_f1 <= 1.0
    assert oof_preds.shape == (len(y),)


def test_works_with_catboost(dummy_data):
    X, y = dummy_data
    params = {"iterations": 50, "verbose": 0}
    mean_f1, oof_preds, oof_proba = cv_score_model_with_earlystop(
        CatBoostClassifier, params, X, y, n_splits=3
    )
    assert 0.0 <= mean_f1 <= 1.0
    assert oof_preds.shape == (len(y),)


# ── edge case tests ──────────────────────────────────────────────────────────

def test_no_proba_returns_none(dummy_data):
    """A model without predict_proba should return None for oof_proba."""
    from sklearn.svm import SVC
    X, y = dummy_data
    # SVC with probability=False has no predict_proba
    _, _, oof_proba = cv_score_model_with_earlystop(
        SVC, {"probability": False}, X, y, n_splits=3
    )
    assert oof_proba is None


def test_different_n_splits(dummy_data):
    X, y = dummy_data
    for n_splits in [3, 5]:
        mean_f1, oof_preds, _ = cv_score_model_with_earlystop(
            LogisticRegression, {}, X, y, n_splits=n_splits
        )
        assert oof_preds.shape == (len(y),)
        assert 0.0 <= mean_f1 <= 1.0




##✅✅

optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence Optuna logs in tests

# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_data():
    """Small balanced 7-class dataset mirroring the obesity dataset."""
    X, y = make_classification(
        n_samples=210,
        n_features=10,
        n_classes=7,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    return X, y


@pytest.fixture
def single_trial(dummy_data):
    """Run one Optuna trial and return the score."""
    X, y = dummy_data
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=1)
    return study.best_value, study.best_trial


# ── return value tests ────────────────────────────────────────────────────────

def test_objective_returns_float(dummy_data):
    X, y = dummy_data
    trial = optuna.trial.create_trial(
        params={
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1e-5,
            "reg_lambda": 1e-5,
            "scale_pos_weight": 1.0,
        },
        distributions={
            "n_estimators": optuna.distributions.IntDistribution(100, 1300),
            "learning_rate": optuna.distributions.FloatDistribution(1e-3, 0.2, log=True),
            "max_depth": optuna.distributions.IntDistribution(3, 9),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
            "colsample_bytree": optuna.distributions.FloatDistribution(0.4, 1.0),
            "reg_alpha": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
            "reg_lambda": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
            "scale_pos_weight": optuna.distributions.CategoricalDistribution([1.0]),
        },
        value=0.5,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective_xgb(t, X, y), n_trials=1)
    assert isinstance(study.best_value, float)


def test_score_between_0_and_1(single_trial):
    best_value, _ = single_trial
    assert 0.0 <= best_value <= 1.0, f"F1 score {best_value} out of range [0, 1]"


# ── optuna integration tests ──────────────────────────────────────────────────

def test_study_maximizes_score(dummy_data):
    """Score should improve or stay equal over multiple trials."""
    X, y = dummy_data
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=3)
    assert study.best_value >= study.trials[0].value


def test_correct_number_of_trials(dummy_data):
    X, y = dummy_data
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=3)
    assert len(study.trials) == 3


def test_all_trials_completed(dummy_data):
    X, y = dummy_data
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=3)
    for trial in study.trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE, \
            f"Trial {trial.number} did not complete: {trial.state}"

##✅✅        


# ── hyperparameter boundary tests for XGB ─────────────────────────────────────────────

optuna.logging.set_verbosity(optuna.logging.WARNING)
def test_sampled_params_within_bounds(dummy_data):
    """Verify Optuna respects the defined search space boundaries."""
    X, y = dummy_data
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=3)

    for trial in study.trials:
        p = trial.params
        assert 100 <= p["n_estimators"] <= 1300
        assert 1e-3 <= p["learning_rate"] <= 0.2
        assert 3   <= p["max_depth"]     <= 9
        assert 0.5 <= p["subsample"]     <= 1.0
        assert 0.4 <= p["colsample_bytree"] <= 1.0
        assert 1e-8 <= p["reg_alpha"]    <= 10.0
        assert 1e-8 <= p["reg_lambda"]   <= 10.0
        assert p["scale_pos_weight"]     == 1.0  # categorical fixed to 1.0


def test_best_trial_has_all_params(single_trial):
    """Best trial must contain all expected hyperparameter keys."""
    _, best_trial = single_trial
    expected_keys = {
        "n_estimators", "learning_rate", "max_depth",
        "subsample", "colsample_bytree", "reg_alpha",
        "reg_lambda", "scale_pos_weight"
    }
    assert expected_keys.issubset(set(best_trial.params.keys()))



optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_data():
    X, y = make_classification(
        n_samples=210,
        n_features=10,
        n_classes=7,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)]), y

##✅✅

# ── hyperparameter boundary tests for lgb ───────────────────────────────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)

def test_objective_lgb_returns_float(dummy_data):
    X, y = dummy_data
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective_lgb(t, X, y), n_trials=1)
    assert isinstance(study.best_value, float)


def test_score_between_0_and_1(dummy_data):
    X, y = dummy_data
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective_lgb(t, X, y), n_trials=1)
    assert 0.0 <= study.best_value <= 1.0


def test_all_trials_complete(dummy_data):
    X, y = dummy_data
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective_lgb(t, X, y), n_trials=3)
    for trial in study.trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE


# ── hyperparameter boundary tests ─────────────────────────────────────────────

def test_sampled_params_within_bounds(dummy_data):
    X, y = dummy_data
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective_lgb(t, X, y), n_trials=3)

    for trial in study.trials:
        p = trial.params
        assert 100  <= p["n_estimators"]      <= 1100
        assert 1e-3 <= p["learning_rate"]     <= 0.2
        assert 15   <= p["num_leaves"]        <= 130
        assert 12   <= p["min_child_samples"] <= 75
        assert 0.0  <= p["min_split_gain"]    <= 0.1
        assert 0.6  <= p["subsample"]         <= 1.0
        assert 0.5  <= p["colsample_bytree"]  <= 1.0
        assert 1e-8 <= p["reg_alpha"]         <= 10.0
        assert 1e-8 <= p["reg_lambda"]        <= 10.0
        assert p["scale_pos_weight"]          == 1.0


def test_best_trial_has_all_params(dummy_data):
    X, y = dummy_data
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective_lgb(t, X, y), n_trials=1)
    expected_keys = {
        "n_estimators", "learning_rate", "num_leaves",
        "min_child_samples", "min_split_gain", "subsample",
        "colsample_bytree", "reg_alpha", "reg_lambda", "scale_pos_weight"
    }
    assert expected_keys.issubset(set(study.best_trial.params.keys()))
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── fixture ───────────────────────────────────────────────────────────────────

##✅✅

@pytest.fixture
def dummy_data():
    X, y = make_classification(
        n_samples=210,
        n_features=10,
        n_classes=7,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)]), y


# ── test optuna function  ───────────────────────────────────────────────────────────────

def test_returns_tuple(dummy_data):
    X, y = dummy_data
    result = run_optuna_tuning(X, y, model_name="LightGBM", n_trials=1)
    assert isinstance(result, tuple) and len(result) == 2


def test_best_params_is_dict(dummy_data):
    X, y = dummy_data
    best_params, _ = run_optuna_tuning(X, y, model_name="LightGBM", n_trials=1)
    assert isinstance(best_params, dict)


def test_best_value_is_float(dummy_data):
    X, y = dummy_data
    _, best_value = run_optuna_tuning(X, y, model_name="LightGBM", n_trials=1)
    assert isinstance(best_value, float)


def test_best_value_between_0_and_1(dummy_data):
    X, y = dummy_data
    _, best_value = run_optuna_tuning(X, y, model_name="LightGBM", n_trials=1)
    assert 0.0 <= best_value <= 1.0


# ── model name tests ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", ["XGBoost", "LightGBM", "CatBoost"])
def test_all_models_run(dummy_data, model_name):
    X, y = dummy_data
    best_params, best_value = run_optuna_tuning(X, y, model_name=model_name, n_trials=1)
    assert isinstance(best_params, dict)
    assert 0.0 <= best_value <= 1.0


def test_invalid_model_raises_error(dummy_data):
    X, y = dummy_data
    with pytest.raises(ValueError):
        run_optuna_tuning(X, y, model_name="RandomForest", n_trials=1)


# ── n_trials tests ────────────────────────────────────────────────────────────

def test_n_trials_respected(dummy_data):
    """Best params should be non-empty dict after n trials."""
    X, y = dummy_data
    best_params, _ = run_optuna_tuning(X, y, model_name="LightGBM", n_trials=2)
    assert len(best_params) > 0


def test_more_trials_returns_valid_result(dummy_data):
    X, y = dummy_data
    best_params, best_value = run_optuna_tuning(X, y, model_name="LightGBM", n_trials=3)
    assert isinstance(best_params, dict)
    assert 0.0 <= best_value <= 1.0

##✅✅

@pytest.fixture
def dummy_classification():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
    labels = [0, 1, 2]
    return y_true, y_pred, labels


# ── basic tests ───────────────────────────────────────────────────────────────

def test_file_is_created(dummy_classification, tmp_path):
    y_true, y_pred, labels = dummy_classification
    outpath = tmp_path / "confusion_matrix.png"
    plot_and_save_confusion(y_true, y_pred, labels, "Test CM", str(outpath))
    assert outpath.exists()


def test_file_is_not_empty(dummy_classification, tmp_path):
    y_true, y_pred, labels = dummy_classification
    outpath = tmp_path / "confusion_matrix.png"
    plot_and_save_confusion(y_true, y_pred, labels, "Test CM", str(outpath))
    assert outpath.stat().st_size > 0


def test_output_is_png(dummy_classification, tmp_path):
    y_true, y_pred, labels = dummy_classification
    outpath = tmp_path / "confusion_matrix.png"
    plot_and_save_confusion(y_true, y_pred, labels, "Test CM", str(outpath))
    # PNG files start with this magic header
    with open(outpath, "rb") as f:
        header = f.read(8)
    assert header[:4] == b'\x89PNG'


def test_no_open_figures_after_call(dummy_classification, tmp_path):
    """plt.close(fig) should be called — no memory leaks."""
    import matplotlib.pyplot as plt
    y_true, y_pred, labels = dummy_classification
    outpath = tmp_path / "confusion_matrix.png"
    before = len(plt.get_fignums())
    plot_and_save_confusion(y_true, y_pred, labels, "Test CM", str(outpath))
    after = len(plt.get_fignums())
    assert after == before


def test_works_with_7_classes(tmp_path):
    """Mirror the actual obesity dataset with 7 classes."""
    y_true = np.tile(np.arange(7), 3)
    y_pred = np.tile(np.arange(7), 3)
    labels = list(range(7))
    outpath = tmp_path / "confusion_7classes.png"
    plot_and_save_confusion(y_true, y_pred, labels, "7 Classes", str(outpath))
    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_perfect_predictions(dummy_classification, tmp_path):
    """Perfect y_true == y_pred should still save without errors."""
    y_true, _, labels = dummy_classification
    outpath = tmp_path / "perfect.png"
    plot_and_save_confusion(y_true, y_true, labels, "Perfect", str(outpath))
    assert outpath.exists()

##✅✅   

# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_roc_data():
    """3-class classification with probability scores."""
    np.random.seed(42)
    n_samples = 90
    y_true  = np.tile(np.arange(3), n_samples // 3)
    # generate soft probabilities that sum to 1
    raw     = np.random.dirichlet(np.ones(3), size=n_samples)
    y_proba = raw
    labels  = [0, 1, 2]
    return y_true, y_proba, labels


@pytest.fixture
def dummy_roc_data_7():
    """7-class version mirroring the obesity dataset."""
    np.random.seed(42)
    n_samples = 210
    y_true  = np.tile(np.arange(7), n_samples // 7)
    y_proba = np.random.dirichlet(np.ones(7), size=n_samples)
    labels  = list(range(7))
    return y_true, y_proba, labels


# ── basic tests ───────────────────────────────────────────────────────────────

def test_file_is_created(dummy_roc_data, tmp_path):
    y_true, y_proba, labels = dummy_roc_data
    outpath = tmp_path / "roc.png"
    plot_and_save_roc(y_true, y_proba, labels, "Test ROC", str(outpath))
    assert outpath.exists()


def test_file_is_not_empty(dummy_roc_data, tmp_path):
    y_true, y_proba, labels = dummy_roc_data
    outpath = tmp_path / "roc.png"
    plot_and_save_roc(y_true, y_proba, labels, "Test ROC", str(outpath))
    assert outpath.stat().st_size > 0


def test_output_is_png(dummy_roc_data, tmp_path):
    y_true, y_proba, labels = dummy_roc_data
    outpath = tmp_path / "roc.png"
    plot_and_save_roc(y_true, y_proba, labels, "Test ROC", str(outpath))
    with open(outpath, "rb") as f:
        header = f.read(4)
    assert header == b'\x89PNG'


def test_no_open_figures_after_call(dummy_roc_data, tmp_path):
    """plt.close(fig) must be called — no memory leaks."""
    y_true, y_proba, labels = dummy_roc_data
    outpath = tmp_path / "roc.png"
    before = len(plt.get_fignums())
    plot_and_save_roc(y_true, y_proba, labels, "Test ROC", str(outpath))
    after = len(plt.get_fignums())
    assert after == before


# ── data shape tests ──────────────────────────────────────────────────────────

def test_works_with_7_classes(dummy_roc_data_7, tmp_path):
    y_true, y_proba, labels = dummy_roc_data_7
    outpath = tmp_path / "roc_7classes.png"
    plot_and_save_roc(y_true, y_proba, labels, "7 Classes ROC", str(outpath))
    assert outpath.exists()
    assert outpath.stat().st_size > 0


def test_proba_shape_matches_labels(dummy_roc_data):
    """y_proba columns must match number of labels."""
    y_true, y_proba, labels = dummy_roc_data
    assert y_proba.shape[1] == len(labels)


def test_proba_rows_sum_to_one(dummy_roc_data):
    """Valid probability distribution must sum to 1 per sample."""
    _, y_proba, _ = dummy_roc_data
    np.testing.assert_allclose(y_proba.sum(axis=1), np.ones(len(y_proba)), atol=1e-6)


# ── edge case tests ───────────────────────────────────────────────────────────

def test_perfect_proba(tmp_path):
    """Perfect classifier should produce AUC=1.0 without errors."""
    labels  = [0, 1, 2]
    y_true  = np.array([0, 1, 2, 0, 1, 2])
    y_proba = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    outpath = tmp_path / "roc_perfect.png"
    plot_and_save_roc(y_true, y_proba, labels, "Perfect ROC", str(outpath))
    assert outpath.exists()


def test_different_output_paths(dummy_roc_data, tmp_path):
    """Function should save to whatever path is given."""
    y_true, y_proba, labels = dummy_roc_data
    for name in ["roc_1.png", "roc_2.png", "roc_3.png"]:
        outpath = tmp_path / name
        plot_and_save_roc(y_true, y_proba, labels, name, str(outpath))
        assert outpath.exists()

##✅✅         
# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_split():
    X, y = make_classification(
        n_samples=350, n_features=10, n_classes=7,
        n_informative=8, n_redundant=2, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


@pytest.fixture
def minimal_params():
    return {
        "XGBoost":  {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
        "LightGBM": {"n_estimators": 50, "num_leaves": 15, "verbose": -1},
        "CatBoost": {"iterations": 50, "depth": 3},
    }


# ── fit_final_and_evaluate ────────────────────────────────────────────────────

@pytest.fixture
def results(dummy_split, minimal_params, tmp_path):
    """Run once, reuse across tests."""
    X_train, X_test, y_train, y_test = dummy_split
    with patch("src.evaluate.PLOTS_DIR", str(tmp_path)), \
         patch("src.evaluate.MODELS_DIR", str(tmp_path)):
        return fit_final_and_evaluate(X_train, y_train, X_test, y_test, minimal_params)


def test_returns_all_models(results):
    assert set(results.keys()) == {"XGBoost", "LightGBM", "CatBoost"}


def test_all_metric_keys_present(results):
    expected = {
        "model", "accuracy", "f1_macro", "precision_macro",
        "recall_macro", "kappa", "log_loss", "roc_auc_ovr",
        "classification_report", "confusion_matrix"
    }
    for name in results:
        assert expected.issubset(results[name].keys())


@pytest.mark.parametrize("metric", ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr"])
def test_metrics_in_valid_range(results, metric):
    for name in results:
        assert 0.0 <= results[name][metric] <= 1.0, f"{name} {metric} out of range"


def test_kappa_in_valid_range(results):
    for name in results:
        assert -1.0 <= results[name]["kappa"] <= 1.0


def test_log_loss_is_positive(results):
    for name in results:
        assert results[name]["log_loss"] > 0


def test_confusion_matrix_shape(results, dummy_split):
    _, X_test, _, y_test = dummy_split
    n_classes = len(np.unique(y_test))
    for name in results:
        assert results[name]["confusion_matrix"].shape == (n_classes, n_classes)


def test_unknown_model_skipped(dummy_split, tmp_path):
    X_train, X_test, y_train, y_test = dummy_split
    with patch("src.evaluate.PLOTS_DIR", str(tmp_path)), \
         patch("src.evaluate.MODELS_DIR", str(tmp_path)):
        results = fit_final_and_evaluate(
            X_train, y_train, X_test, y_test,
            {"UnknownModel": {"n_estimators": 50}}
        )
    assert "UnknownModel" not in results


# ── main ──────────────────────────────────────────────────────────────────────

def test_main_returns_results(tmp_path):
    """Main should complete and return a results dict with all 3 models."""
    with patch("src.evaluate.PLOTS_DIR", str(tmp_path)), \
         patch("src.evaluate.MODELS_DIR", str(tmp_path)), \
         patch("src.evaluate.run_optuna_tuning") as mock_tune, \
         patch("src.evaluate.fit_final_and_evaluate") as mock_fit:

        mock_tune.return_value = ({"n_estimators": 50}, 0.85)
        mock_fit.return_value = {
            "XGBoost":  {"f1_macro": 0.85},
            "LightGBM": {"f1_macro": 0.87},
            "CatBoost": {"f1_macro": 0.86},
        }

        results = main(data_path="fake.csv", target_col="NObeyesdad", n_trials=1)
        assert isinstance(results, dict)
        assert mock_tune.call_count == 3   # called once per model
        assert mock_fit.call_count == 1