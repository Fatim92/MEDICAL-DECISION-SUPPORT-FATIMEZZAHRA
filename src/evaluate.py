import os
import numpy as np
import pandas as pd
import joblib
import optuna
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, cohen_kappa_score, confusion_matrix,
    classification_report, roc_curve, auc, ConfusionMatrixDisplay
# for Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED) 
##Creating 2 new folder were to put plots(graphs) and model
PLOTS_DIR = "plots"
MODELS_DIR = "models"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
def load_data(filepath="data_path", target_col="NObeyesdad"):
    df = pd.read_csv(filepath)
    df = df.reset_index(drop=True)
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    y=y.values.ravel()
    return X, y
## Defining predection and meanF1 over KFold function
def cv_score_model_with_earlystop(model_ctor, params, X, y, n_splits=5, early_stopping_rounds=42):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    f1_scores = []
    oof_proba = None
    oof_preds = np.zeros(len(y), dtype=int)
    # if predict_proba available we gather probabilities into array
    supports_proba = True
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = model_ctor(**params)

        # boosters: XGB, LGB, CatBoost -> have early stopping
        if isinstance(model, XGBClassifier):
            model=XGBClassifier(**params,early_stopping_rounds=early_stopping_rounds)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        elif isinstance(model, lgb.LGBMClassifier):
           model.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(42), lgb.log_evaluation(False)])
        elif isinstance(model, CatBoostClassifier):
            model.fit(X_tr, y_tr,
                      eval_set=(X_val, y_val),
                      early_stopping_rounds=early_stopping_rounds,
                      verbose=False)
        else:
            model.fit(X_tr, y_tr)

        # predictions
        y_pred=model.predict(X_val) # model predection 
        y_pred = y_pred.reshape(-1)# guarantee 1D output
        oof_preds[val_idx] = y_pred
        f1_scores.append(f1_score(y_val, y_pred, average='macro'))

        # probabilities
        if hasattr(model, "predict_proba"):#test if model has a methode predict_proba to avoid crashing 
            probs = model.predict_proba(X_val)
            if oof_proba is None:
                oof_proba = np.zeros((len(y), probs.shape[1]))
            oof_proba[val_idx, :] = probs
        else:
            supports_proba = False

    mean_f1 = float(np.mean(f1_scores)) #calculating mean of F1 Score over K-Folds
    return mean_f1, oof_preds, (oof_proba if supports_proba else None)


def objective_xgb(trial, X, y):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1300),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0]),
        "random_state": RANDOM_SEED,
        "use_label_encoder": False,
        "verbosity": 0,
        "n_jobs": -1,
        "eval_metric": "mlogloss"
    }
    def ctor(**p): return XGBClassifier(**p)
    score, oof_preds, oof_proba = cv_score_model_with_earlystop(ctor, params, X, y, n_splits=5, early_stopping_rounds=50)
    return score

def objective_lgb(trial, X, y):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1100),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 15, 130),
        "min_child_samples": trial.suggest_int("min_child_samples", 12, 75),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
        "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "n_jobs": -1,
    }
    params["scale_pos_weight"] = trial.suggest_categorical("scale_pos_weight", [1.0])
    def ctor(**p): return lgb.LGBMClassifier(**p)
    score, oof_preds, oof_proba = cv_score_model_with_earlystop(ctor, params, X, y, n_splits=5, early_stopping_rounds=50)
    return score

def run_optuna_tuning(X, y, model_name='LightGBM', n_trials=48):
    if model_name == 'XGBoost':
        func = lambda trial: objective_xgb(trial, X, y)
    elif model_name == 'LightGBM':
        func = lambda trial: objective_lgb(trial, X, y)
    elif model_name == 'CatBoost':
        func = lambda trial: objective_cat(trial, X, y)
    else:
        raise ValueError(model_name)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(func, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    print(f"[{model_name}] Best value: {study.best_value:.5f}")
    return study.best_params, study.best_value

def plot_and_save_confusion(y_true, y_pred, labels, title, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_and_save_roc(y_true, y_proba, labels, title, outpath):
    # y_proba: shape = (n_samples, n_classes)
    y_true_bin = label_binarize(y_true, classes=labels)
    n_classes = y_proba.shape[1]
    # compute per-class ROC
    fig, ax = plt.subplots(figsize=(7,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"Class {labels[i]} (AUC={roc_auc:.3f})")
    # micro-average
    try:
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        ax.plot(fpr, tpr, label=f"Micro (AUC={auc(fpr,tpr):.3f})", linestyle="--", color="black")
    except Exception:
        pass
    ax.plot([0,1],[0,1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)



from lightgbm.callback import early_stopping
def fit_final_and_evaluate(X_train, y_train, X_test, y_test, best_params_per_model):
    results = {}
    label_list = np.unique(y_test).tolist()

    for name, params in best_params_per_model.items():
        print(f"\nFitting final model: {name}")
        p = params.copy()
        p.setdefault("random_state", RANDOM_SEED)

        if name == "XGBoost":
            model = XGBClassifier(**p, use_label_encoder=False, verbosity=0,early_stopping_rounds=40)
        elif name == "LightGBM":
            model = lgb.LGBMClassifier(**p)
        elif name == "CatBoost":
            model = CatBoostClassifier(**p, verbose=False)
        else:
            continue

        # fit with early stopping
        if isinstance(model, XGBClassifier):
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        elif isinstance(model, lgb.LGBMClassifier):
            model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          callbacks=[lgb.early_stopping(42), lgb.log_evaluation(False)])
        elif isinstance(model, CatBoostClassifier):
            model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=40, verbose=False)
        else:
            model.fit(X_train, y_train)

        # predictions & metrics
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")
        kappa = cohen_kappa_score(y_test, y_pred)
        ll = log_loss(y_test, y_proba) if y_proba is not None else None
        roc_auc_ovr = None
        if y_proba is not None:
            y_test_bin = label_binarize(y_test, classes=label_list)
            roc_auc_ovr = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")

        # save results, print
        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1_macro": f1m,
            "precision_macro": prec,
            "recall_macro": rec,
            "kappa": kappa,
            "log_loss": ll,
            "roc_auc_ovr": roc_auc_ovr,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }

        print(f"{name} -> F1-macro: {f1m:.4f}, Acc: {acc:.4f}, Kappa: {kappa:.4f}, ROC_AUC(ovr): {roc_auc_ovr}")

        # plots
        cm_path = os.path.join(PLOTS_DIR, f"{name}_confusion.png")
        roc_path = os.path.join(PLOTS_DIR, f"{name}_roc.png")
        plot_and_save_confusion(y_test, y_pred, labels=label_list, title=f"{name} Confusion Matrix", outpath=cm_path)
        if y_proba is not None:
            plot_and_save_roc(y_test, y_proba, labels=label_list, title=f"{name} ROC (OvR)", outpath=roc_path)
        # save model
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}_final.joblib"))

        # print classification report (nice formatted)
        print("\nClassification report:\n", classification_report(y_test, y_pred))

    return results


def main(data_path="/content/processed_data (1).csv", target_col="NObeyesdad", n_trials=36):
    X, y = load_data(data_path, target_col)

    # final holdout
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

    models_to_tune = ["XGBoost", "LightGBM", "CatBoost"]
    best_params = {}
    best_scores = {}

    for m in models_to_tune:
        print(f"\n==== Tuning {m} with Optuna ({n_trials} trials) ====")
        params, score = run_optuna_tuning(X_train_full, y_train_full, model_name=m, n_trials=n_trials)
        best_params[m] = params
        best_scores[m] = score

    print("\nBest params summary:")
    for k,v in best_params.items():
        print(f" - {k}: {v}")

    results = fit_final_and_evaluate(X_train_full, y_train_full, X_test, y_test, best_params)