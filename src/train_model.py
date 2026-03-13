import os
import joblib
import numpy as np
import pandas as pd
import optuna

from typing import List, Tuple
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils import check_array

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

## based models path
MODEL_PATHS = [
    "/content/models/XGBoost_final.joblib",
    "/content/models/LightGBM_final.joblib",
    "/content/models/CatBoost_final.joblib"
]
NEW_DATA_CSV = "Data/ObesityDataSet_raw_and_data_sinthetic.csv"   ##data for training
OUT_PIPELINE_PATH = "/content/models/logistic_meta_minimal.joblib" ## output path for the joblib model file
N_TRIALS = 30
CV_SPLITS = 5
# ============================================================

def load_base_models(paths: List[str]) -> List[Tuple[str, object]]: ##load base models  to return list tuple name and model
    
    loaded = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model file not found: {p}")
        name = os.path.splitext(os.path.basename(p))[0]
        model = joblib.load(p)
        loaded.append((name, model))
    return loaded

def build_meta_features(models: List[Tuple[str, object]], X: pd.DataFrame):
   
    parts = []
    names = []
    pred_classes_list = []

    for name, m in models:
        # predict_proba preferred
        if hasattr(m, "predict_proba"):
            probs = np.asarray(m.predict_proba(X))
            if probs.ndim == 1:
                probs = probs.reshape(-1, 1)
            parts.append(probs)
            for j in range(probs.shape[1]):
                names.append(f"{name}_p{j}")
            preds = np.asarray(m.predict(X)).ravel()
            parts.append(preds.reshape(-1,1)); names.append(f"{name}_pred")
            maxp = probs.max(axis=1).reshape(-1,1)
            parts.append(maxp); names.append(f"{name}_maxp")
            pred_classes_list.append(preds)
        else:
            preds = np.asarray(m.predict(X)).ravel()
            parts.append(preds.reshape(-1,1)); names.append(f"{name}_pred")
            maxp = np.ones((X.shape[0],1))
            parts.append(maxp); names.append(f"{name}_maxp")
            pred_classes_list.append(preds)

    # agreement count (how many models agree on mode)
    if len(pred_classes_list) >= 2:
        arr = np.vstack(pred_classes_list)  
        # compute mode count per sample
        from scipy.stats import mode
        mvals, _ = mode(arr, axis=0)
        mode_counts = (arr == mvals).sum(axis=0).reshape(-1,1)
        parts.append(mode_counts); names.append("models_agreement")
    else:
        parts.append(np.ones((X.shape[0],1))); names.append("models_agreement")

    meta = np.hstack(parts)
    return meta, names

def objective_sgd(trial, X_meta, y, cv_splits=CV_SPLITS): ## parameters to search  and spliting data to fit the model
    # minimal search space tuned for stability
    lr = trial.suggest_categorical("learning_rate", ["constant","optimal","invscaling","adaptive"])
    eta0 = trial.suggest_float("eta0", 1e-4, 1.0, log=True) if lr in ("constant","invscaling","adaptive") else 0.0  
    alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)                                                       
    max_iter = trial.suggest_int("max_iter", 1000, 3000)

    clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, learning_rate=lr,                                   
                      eta0=eta0 if eta0>0 else 0.0, max_iter=max_iter, random_state=RANDOM_SEED, tol=1e-4)
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for tr_idx, val_idx in cv.split(X_meta, y):
        X_tr, X_val = X_meta[tr_idx], X_meta[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        scores.append(f1_score(y_val, preds, average="macro"))
    return float(np.mean(scores))

def main():## findin best params for meta model with optuna and fiting the model/saving
    #  file checks
    if not os.path.exists(NEW_DATA_CSV):
        raise FileNotFoundError(f"New data file not found: {NEW_DATA_CSV}")

    # load base models
    models = load_base_models(MODEL_PATHS)

    # load dataset
    df = pd.read_csv(NEW_DATA_CSV)
    target_col="NObeyesdad"
    X = df.drop(columns=[target_col])
    y = df[target_col].values.reshape(-1)

    # ensure base models can predict a small batch 
    try:
        _ = [m.predict(X.iloc[:5]) for _, m in models]
    except Exception as e:
        raise RuntimeError(f"Base model prediction failed on new data. Ensure feature order/types match. Error: {e}")

    # build meta-features from predection of base models
    meta_X, feature_names = build_meta_features(models, X)
    meta_y = y.copy()

    # optuna tuning 
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda t: objective_sgd(t, meta_X, meta_y, CV_SPLITS), n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

    best = study.best_params
    best_score = study.best_value
    print("Optuna best score (F1-macro):", best_score)
    print("Optuna best params:", best)

    # train final pipeline on whole meta data with best params
    lr = best["learning_rate"]
    eta0 = best.get("eta0", 0.0) if lr in ("constant","invscaling","adaptive") else 0.0
    alpha = best["alpha"]
    max_iter = best.get("max_iter", 1000)

    final_pipe = make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, learning_rate=lr,                                   # fix: loss="log" deprecated
                      eta0=eta0 if eta0>0 else 0.0, max_iter=max_iter, random_state=RANDOM_SEED, tol=1e-4)
    )
    final_pipe.fit(meta_X, meta_y)

    # save final pipeline and metadata
    os.makedirs(os.path.dirname(OUT_PIPELINE_PATH) or ".", exist_ok=True)
    joblib.dump({
        "pipeline": final_pipe,
        "base_models": MODEL_PATHS,
        "meta_feature_names": feature_names,
        "optuna_best_params": best,
        "optuna_best_score": best_score
    }, OUT_PIPELINE_PATH)

    print("Saved logistic meta-pipeline to:", OUT_PIPELINE_PATH)

if __name__ == "__main__":
    main()