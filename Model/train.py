#!/usr/bin/env python
"""
train.py — Treinamento do modelo de detecção de fraudes com Optuna (F-beta otimização) + Cross-Validation.

Pipeline:
1. Carrega ABT (Dados/abt.csv)
2. Split treino/validação/teste
3. Compara 3 modelos vanilla (XGBoost, LightGBM, RandomForest) com F-beta(beta=2)
4. Seleciona o melhor modelo
5. Otimiza hiperparâmetros do melhor modelo via Optuna (maximizando F-beta beta=2)
6. Cross-Validation estratificada para detectar overfitting
7. Treino final no conjunto treino+validação
8. Avaliação no teste + salvamento

Uso:
    python Model/train.py [--trials N] [--timeout SEC] [--no-optuna] [--cv-folds N]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
)
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier

# Verificar XGBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERRO: xgboost não instalado. Execute: pip install xgboost")
    sys.exit(1)

# Verificar LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("AVISO: lightgbm não instalado. Execute: pip install lightgbm")

# Optuna
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("AVISO: optuna não instalado. Execute: pip install optuna")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carregamento de configuração
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("model_config.json")

with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    CFG = json.load(fh)

MODEL_CFG = CFG["model"]
HYPERPARAMS = CFG["hyperparameters"]
TRAIN_CFG = CFG["training"]
EVAL_CFG = CFG["evaluation"]

# Paths
ABT_PATH = Path(__file__).parent.parent / "Dados" / "abt.csv"
MODEL_OUTPUT = Path(__file__).parent / "fraud_model.pkl"
METRICS_OUTPUT = Path(__file__).parent / "training_metrics.json"
OPTUNA_STUDY_OUTPUT = Path(__file__).parent / "optuna_study.pkl"


# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------
def json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(json_serializable(v) for v in obj)
    return obj


def load_abt() -> pd.DataFrame:
    """Carrega a ABT."""
    logger.info("Carregando ABT: %s", ABT_PATH)
    df = pd.read_csv(ABT_PATH)
    logger.info("ABT carregada: %s linhas x %s colunas", f"{df.shape[0]:,}", df.shape[1])
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """Separa treino, validação e teste."""
    target_col = TRAIN_CFG["target_column"]
    test_size = TRAIN_CFG["test_size"]
    random_state = TRAIN_CFG["random_state"]
    stratify = TRAIN_CFG["stratify"]
    val_split = TRAIN_CFG["validation_split"]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Primeiro split: treino+val / teste
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )

    # Segundo split: treino / validação (para early stopping e Optuna)
    val_size = val_split / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size,
        random_state=random_state, stratify=y_trainval if stratify else None
    )

    logger.info(
        "Split: train=%s, val=%s, test=%s",
        f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}"
    )
    logger.info(
        "Fraude - train: %.2f%%, val: %.2f%%, test: %.2f%%",
        y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Modelos Vanilla para Comparação Inicial
# ---------------------------------------------------------------------------
def get_vanilla_models(random_state=42):
    """Retorna dicionário com 3 modelos vanilla (sem tuning)."""
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=25.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="auc",
            tree_method="hist",
            verbosity=0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if LGB_AVAILABLE:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=25.0,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True,
        )
    return models


def evaluate_vanilla_models(models: dict, X_train, y_train, X_val, y_val) -> dict:
    """Avalia modelos vanilla no validation set e retorna métricas + melhor modelo."""
    logger.info("=" * 60)
    logger.info("COMPARAÇÃO DE MODELOS VANILLA (F-beta beta=2)")
    logger.info("=" * 60)

    results = {}
    best_model_name = None
    best_fbeta = -1

    for name, model in models.items():
        logger.info("Treinando %s...", name)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        fbeta = fbeta_score(y_val, y_pred, beta=2.0)
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        results[name] = {
            "model": model,
            "fbeta_2": float(fbeta),
            "roc_auc": float(roc_auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        logger.info(
            "  %s: F-beta(2)=%.4f, ROC-AUC=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f",
            name, fbeta, roc_auc, precision, recall, f1
        )

        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_model_name = name

    logger.info("=" * 60)
    logger.info("MELHOR MODELO VANILLA: %s (F-beta=%.4f)", best_model_name, best_fbeta)
    logger.info("=" * 60)

    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_fbeta": best_fbeta,
    }


# ---------------------------------------------------------------------------
# Optuna Objective - Maximizar F-beta (beta=2) para cada modelo
# ---------------------------------------------------------------------------
def create_optuna_objective(model_name: str, X_train, y_train, X_val, y_val, random_state=42):
    """Cria função objective para Optuna otimizando F-beta (beta=2) para o modelo específico."""

    def objective(trial):
        if model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 10, 50),
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": 0,
            }
            model = XGBClassifier(**params)

        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 10, 50),
                "objective": "binary",
                "metric": "auc",
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": -1,
                "force_col_wise": True,
            }
            model = lgb.LGBMClassifier(**params)

        elif model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
                "random_state": random_state,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)

        else:
            raise ValueError(f"Modelo desconhecido: {model_name}")

        # Treino
        if model_name in ["XGBoost", "LightGBM"]:
            # Early stopping para gradient boosting
            if hasattr(model, "set_params"):
                model.set_params(early_stopping_rounds=50)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        # Avaliar no validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # F-beta score com beta=2 (prioriza recall)
        fbeta = fbeta_score(y_val, y_pred, beta=2.0)

        return fbeta

    return objective


def run_optuna(model_name: str, X_train, y_train, X_val, y_val, n_trials=30, timeout=600):
    """Executa otimização Optuna para o modelo especificado."""
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna não disponível, usando hiperparâmetros padrão")
        return {}

    logger.info("=" * 60)
    logger.info("OTIMIZAÇÃO OPTUNA - %s (Maximizando F-beta beta=2)", model_name)
    logger.info("=" * 60)
    logger.info("Trials: %d, Timeout: %ds", n_trials, timeout)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=TRAIN_CFG["random_state"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    objective = create_optuna_objective(model_name, X_train, y_train, X_val, y_val)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    logger.info("Melhor F-beta (beta=2): %.4f", study.best_value)
    logger.info("Melhores parâmetros: %s", study.best_params)

    # Salvar study
    with OPTUNA_STUDY_OUTPUT.open("wb") as fh:
        pickle.dump(study, fh)

    return study.best_params


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------
def run_cross_validation(X_trainval: pd.DataFrame, y_trainval: pd.Series, model_name: str, best_params: dict) -> dict:
    """Executa Cross-Validation estratificada no conjunto treino+validação."""
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION (StratifiedKFold) - %s", model_name)
    logger.info("=" * 60)

    n_splits = TRAIN_CFG.get("cv_folds", 5)
    random_state = TRAIN_CFG["random_state"]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Modelo para CV com melhores parâmetros
    if model_name == "XGBoost":
        cv_params = HYPERPARAMS.copy()
        cv_params.update(best_params)
        cv_params["random_state"] = cv_params.get("random_state", random_state)
        cv_model = XGBClassifier(**cv_params)
    elif model_name == "LightGBM":
        cv_params = HYPERPARAMS.copy()
        cv_params.update(best_params)
        cv_params["random_state"] = cv_params.get("random_state", random_state)
        cv_model = lgb.LGBMClassifier(**cv_params)
    elif model_name == "RandomForest":
        cv_params = HYPERPARAMS.copy() if "RandomForest" in HYPERPARAMS else {}
        cv_params.update(best_params)
        cv_params["random_state"] = cv_params.get("random_state", random_state)
        cv_model = RandomForestClassifier(**cv_params)

    # Métricas para CV
    scoring_metrics = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    cv_results = {}

    for metric_name, scoring in scoring_metrics.items():
        logger.info("Executando CV para %s...", metric_name)
        scores = cross_val_score(
            cv_model, X_trainval, y_trainval,
            cv=cv, scoring=scoring, n_jobs=-1, verbose=0
        )
        cv_results[metric_name] = {
            "scores": [float(s) for s in scores],
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }
        logger.info(
            "  %s: mean=%.4f, std=%.4f, scores=%s",
            metric_name.upper(),
            cv_results[metric_name]["mean"],
            cv_results[metric_name]["std"],
            [f"{s:.4f}" for s in scores],
        )

    # Cross-validated predictions para análise detalhada
    logger.info("Gerando predições CV para análise de curva ROC/PR...")
    y_pred_proba_cv = cross_val_predict(
        cv_model, X_trainval, y_trainval,
        cv=cv, method="predict_proba", n_jobs=-1, verbose=0
    )[:, 1]

    y_pred_cv = (y_pred_proba_cv >= 0.5).astype(int)

    # Métricas agregadas das predições CV
    cv_results["predictions"] = {
        "roc_auc": float(roc_auc_score(y_trainval, y_pred_proba_cv)),
        "precision": float(precision_score(y_trainval, y_pred_cv)),
        "recall": float(recall_score(y_trainval, y_pred_cv)),
        "f1": float(f1_score(y_trainval, y_pred_cv)),
        "fbeta_2": float(fbeta_score(y_trainval, y_pred_cv, beta=2.0)),
    }

    precision_vals, recall_vals, _ = precision_recall_curve(y_trainval, y_pred_proba_cv)
    cv_results["predictions"]["pr_auc"] = float(auc(recall_vals, precision_vals))

    logger.info("CV Predictions - ROC-AUC: %.4f, PR-AUC: %.4f, F1: %.4f, F-beta(2): %.4f",
                cv_results["predictions"]["roc_auc"],
                cv_results["predictions"]["pr_auc"],
                cv_results["predictions"]["f1"],
                cv_results["predictions"]["fbeta_2"])

    return cv_results


def check_overfitting(cv_results: dict, test_metrics: dict) -> dict:
    """Compara performance CV vs Teste para detectar overfitting."""
    logger.info("=" * 60)
    logger.info("CHECK OVERFITTING: CV vs Test")
    logger.info("=" * 60)

    overfitting_report = {}

    # Comparar ROC-AUC
    cv_roc_mean = cv_results["roc_auc"]["mean"]
    test_roc = test_metrics["roc_auc"]
    roc_gap = cv_roc_mean - test_roc
    overfitting_report["roc_auc_gap"] = float(roc_gap)
    overfitting_report["cv_roc_auc_mean"] = float(cv_roc_mean)
    overfitting_report["test_roc_auc"] = float(test_roc)

    logger.info("ROC-AUC: CV_mean=%.4f, Test=%.4f, Gap=%.4f",
                cv_roc_mean, test_roc, roc_gap)

    # Comparar F1
    cv_f1_mean = cv_results["f1"]["mean"]
    test_f1 = test_metrics["f1"]
    f1_gap = cv_f1_mean - test_f1
    overfitting_report["f1_gap"] = float(f1_gap)
    overfitting_report["cv_f1_mean"] = float(cv_f1_mean)
    overfitting_report["test_f1"] = float(test_f1)

    logger.info("F1: CV_mean=%.4f, Test=%.4f, Gap=%.4f",
                cv_f1_mean, test_f1, f1_gap)

    # Comparar F-beta(2)
    cv_fbeta_mean = cv_results["predictions"]["fbeta_2"]
    test_fbeta = test_metrics["fbeta_2"]
    fbeta_gap = cv_fbeta_mean - test_fbeta
    overfitting_report["fbeta_2_gap"] = float(fbeta_gap)
    overfitting_report["cv_fbeta_2_mean"] = float(cv_fbeta_mean)
    overfitting_report["test_fbeta_2"] = float(test_fbeta)

    logger.info("F-beta(2): CV=%.4f, Test=%.4f, Gap=%.4f",
                cv_fbeta_mean, test_fbeta, fbeta_gap)

    # Verificar overfitting
    overfitting_detected = False
    warnings = []

    if roc_gap > 0.02:
        overfitting_detected = True
        warnings.append(f"ROC-AUC gap > 0.02 ({roc_gap:.4f}) - possível overfitting")
        logger.warning("⚠️  ROC-AUC gap > 0.02 (%.4f) - POSSÍVEL OVERFITTING", roc_gap)

    if f1_gap > 0.03:
        overfitting_detected = True
        warnings.append(f"F1 gap > 0.03 ({f1_gap:.4f}) - possível overfitting")
        logger.warning("⚠️  F1 gap > 0.03 (%.4f) - POSSÍVEL OVERFITTING", f1_gap)

    if fbeta_gap > 0.03:
        overfitting_detected = True
        warnings.append(f"F-beta(2) gap > 0.03 ({fbeta_gap:.4f}) - possível overfitting")
        logger.warning("⚠️  F-beta(2) gap > 0.03 (%.4f) - POSSÍVEL OVERFITTING", fbeta_gap)

    # Verificar variância alta no CV (instabilidade)
    cv_roc_std = cv_results["roc_auc"]["std"]
    if cv_roc_std > 0.01:
        warnings.append(f"CV ROC-AUC std alto ({cv_roc_std:.4f}) - modelo instável")
        logger.warning("⚠️  CV ROC-AUC std alto (%.4f) - modelo instável", cv_roc_std)

    overfitting_report["overfitting_detected"] = overfitting_detected
    overfitting_report["warnings"] = warnings

    if not overfitting_detected:
        logger.info("✅ Nenhum sinal forte de overfitting detectado")

    return overfitting_report


# ---------------------------------------------------------------------------
# Treino e Avaliação Final
# ---------------------------------------------------------------------------
def train_final_model(X_trainval: pd.DataFrame, y_trainval: pd.Series, model_name: str, best_params: dict):
    """Treina modelo final com melhores parâmetros em todo o conjunto treino+validação."""
    random_state = TRAIN_CFG["random_state"]

    if model_name == "XGBoost":
        params = HYPERPARAMS.copy()
        params.update(best_params)
        params["random_state"] = params.get("random_state", random_state)
        model = XGBClassifier(**params)
    elif model_name == "LightGBM":
        params = HYPERPARAMS.copy()
        params.update(best_params)
        params["random_state"] = params.get("random_state", random_state)
        model = lgb.LGBMClassifier(**params)
    elif model_name == "RandomForest":
        params = HYPERPARAMS.copy() if "RandomForest" in HYPERPARAMS else {}
        params.update(best_params)
        params["random_state"] = params.get("random_state", random_state)
        model = RandomForestClassifier(**params)

    logger.info("Treinando modelo final (%s) em todo o conjunto treino+validação...", model_name)
    model.fit(X_trainval, y_trainval, verbose=True if model_name == "XGBoost" else False)

    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Avalia o modelo no conjunto de teste."""
    logger.info("Avaliando modelo no conjunto de teste...")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {}

    metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
    metrics["precision"] = float(precision_score(y_test, y_pred))
    metrics["recall"] = float(recall_score(y_test, y_pred))
    metrics["f1"] = float(f1_score(y_test, y_pred))
    metrics["fbeta_2"] = float(fbeta_score(y_test, y_pred, beta=2.0))

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    metrics["pr_auc"] = float(auc(recall_vals, precision_vals))

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    metrics["feature_importance_top20"] = importance.head(20).to_dict(orient="records")

    logger.info("ROC-AUC: %.4f", metrics["roc_auc"])
    logger.info("Precision: %.4f", metrics["precision"])
    logger.info("Recall: %.4f", metrics["recall"])
    logger.info("F1: %.4f", metrics["f1"])
    logger.info("F-beta(2): %.4f", metrics["fbeta_2"])
    logger.info("PR-AUC: %.4f", metrics["pr_auc"])
    logger.info("Confusion Matrix:\n%s", cm)

    return metrics


def save_model(model, model_name: str, metrics: dict, cv_results: dict, overfitting_report: dict, best_params: dict) -> None:
    """Salva o modelo treinado, métricas e resultados de CV."""
    # Modelo
    with MODEL_OUTPUT.open("wb") as fh:
        pickle.dump(model, fh)
    logger.info("Modelo salvo: %s", MODEL_OUTPUT)

    # Métricas completas
    full_metrics = {
        "model_config": CFG,
        "selected_model": model_name,
        "best_params": best_params,
        "test_metrics": metrics,
        "cross_validation": cv_results,
        "overfitting_analysis": overfitting_report,
    }

    full_metrics = json_serializable(full_metrics)

    with METRICS_OUTPUT.open("w", encoding="utf-8") as fh:
        json.dump(full_metrics, fh, indent=2, ensure_ascii=False)
    logger.info("Métricas salvas: %s", METRICS_OUTPUT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30, help="Trials Optuna")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout Optuna (segundos)")
    parser.add_argument("--no-optuna", action="store_true", help="Pula Optuna, usa params padrão")
    parser.add_argument("--cv-folds", type=int, default=5, help="Folds CV")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MODEL TRAINING - Fraud Detection (Vanilla Comparison + Optuna + CV)")
    logger.info("=" * 60)
    logger.info("Target: Maximizar F-beta (beta=2) via Optuna no melhor modelo vanilla")

    # 1. Carregar ABT
    df = load_abt()

    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # 3. Comparar modelos vanilla
    vanilla_models = get_vanilla_models(TRAIN_CFG["random_state"])
    vanilla_results = evaluate_vanilla_models(vanilla_models, X_train, y_train, X_val, y_val)
    best_model_name = vanilla_results["best_model_name"]

    # 4. Optuna Optimization no melhor modelo
    if not args.no_optuna and OPTUNA_AVAILABLE:
        best_params = run_optuna(best_model_name, X_train, y_train, X_val, y_val, args.trials, args.timeout)
    else:
        logger.info("Usando hiperparâmetros padrão do model_config.json")
        best_params = {}

    # 5. Combinar treino + validação para CV e treino final
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)

    # 6. Cross-Validation
    cv_results = run_cross_validation(X_trainval, y_trainval, best_model_name, best_params)

    # 7. Treino final
    logger.info("=" * 60)
    logger.info("TREINAMENTO FINAL (treino + validação) - %s", best_model_name)
    logger.info("=" * 60)
    final_model = train_final_model(X_trainval, y_trainval, best_model_name, best_params)

    # 8. Avaliar no teste
    test_metrics = evaluate_model(final_model, X_test, y_test)

    # 9. Verificar overfitting
    overfitting_report = check_overfitting(cv_results, test_metrics)

    # 10. Salvar
    save_model(final_model, best_model_name, test_metrics, cv_results, overfitting_report, best_params)

    print()
    print("=" * 60)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 60)
    print(f"Modelo Selecionado: {best_model_name}")
    print(f"Test ROC-AUC    : {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC     : {test_metrics['pr_auc']:.4f}")
    print(f"Test F1         : {test_metrics['f1']:.4f}")
    print(f"Test F-beta(2)  : {test_metrics['fbeta_2']:.4f}")
    print(f"Test Precision  : {test_metrics['precision']:.4f}")
    print(f"Test Recall     : {test_metrics['recall']:.4f}")
    print()
    print(f"CV ROC-AUC (mean±std): {cv_results['roc_auc']['mean']:.4f} ± {cv_results['roc_auc']['std']:.4f}")
    print(f"CV F1 (mean±std)       : {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}")
    print(f"CV F-beta(2)           : {cv_results['predictions']['fbeta_2']:.4f}")
    print()
    if overfitting_report["overfitting_detected"]:
        print("⚠️  OVERFITTING DETECTADO:")
        for w in overfitting_report["warnings"]:
            print(f"   - {w}")
    else:
        print("✅ Nenhum overfitting significativo detectado")
    print()
    print(f"Modelo      : {MODEL_OUTPUT}")
    print(f"Métricas    : {METRICS_OUTPUT}")
    print(f"Optuna study: {OPTUNA_STUDY_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()