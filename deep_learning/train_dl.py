#!/usr/bin/env python
"""
train_dl.py — Treinamento de modelo Deep Learning para detecção de fraudes (Keras/TensorFlow).

Usa Keras/TensorFlow para criar uma rede neural profunda para detecção de fraudes,
com foco em maximizar F-beta score (beta=2) para priorizar recall.

Uso:
    python deep_learning/train_dl.py [--epochs N] [--batch-size N] [--lr LR]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Configurar TensorFlow para usar CPU/GPU adequadamente
tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
try:
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent.parent / "Model" / "model_config.json"

with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    CFG = json.load(fh)

TRAIN_CFG = CFG["training"]
EVAL_CFG = CFG["evaluation"]

ABT_PATH = Path(__file__).parent.parent / "Dados" / "abt.csv"
MODEL_OUTPUT = Path(__file__).parent / "fraud_model_dl.keras"
SCALER_OUTPUT = Path(__file__).parent / "scaler_dl.pkl"
METRICS_OUTPUT = Path(__file__).parent / "training_metrics_dl.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modelo Deep Learning (Keras)
# ---------------------------------------------------------------------------
def build_model(input_dim: int, hidden_dims: list = None, dropout: float = 0.3,
                learning_rate: float = 0.001, l2_reg: float = 1e-4) -> keras.Model:
    """Constrói a rede neural para detecção de fraudes."""
    if hidden_dims is None:
        hidden_dims = [512, 256, 128, 64]

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs

    # Camadas ocultas
    for i, hidden_dim in enumerate(hidden_dims):
        x = layers.Dense(
            hidden_dim,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"dense_{i}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    # Camada de saída
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="FraudDetectionNet")

    # Compilar
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )

    return model


# ---------------------------------------------------------------------------
# Callbacks customizados
# ---------------------------------------------------------------------------
class FBetaCallback(callbacks.Callback):
    """Callback para monitorar F-beta score durante treinamento."""
    
    def __init__(self, validation_data, beta=1.75):
        super().__init__()
        self.validation_data = validation_data
        self.beta = beta
        self.best_fbeta = 0.0

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0).flatten()
        
        # Calcular F-beta com threshold 0.5
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        tp = np.sum((y_pred_binary == 1) & (y_val == 1))
        fp = np.sum((y_pred_binary == 1) & (y_val == 0))
        fn = np.sum((y_pred_binary == 0) & (y_val == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        beta_sq = self.beta ** 2
        fbeta = (1 + beta_sq) * precision * (recall) / (beta_sq * precision + recall + 1e-8)
        
        logs = logs or {}
        logs["val_fbeta"] = float(fbeta)
        
        if fbeta > self.best_fbeta:
            self.best_fbeta = float(fbeta)
        
        logger.info(f"Epoch {epoch+1} - val_fbeta: {fbeta:.4f} (best: {self.best_fbeta:.4f})")


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
def load_abt() -> pd.DataFrame:
    logger.info("Carregando ABT: %s", ABT_PATH)
    df = pd.read_csv(ABT_PATH)
    logger.info("ABT: %s linhas x %s colunas", f"{df.shape[0]:,}", df.shape[1])
    return df


def split_data(df: pd.DataFrame):
    target_col = TRAIN_CFG["target_column"]
    test_size = TRAIN_CFG["test_size"]
    random_state = TRAIN_CFG["random_state"]
    stratify = TRAIN_CFG["stratify"]
    val_split = TRAIN_CFG["validation_split"]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )

    val_size = val_split / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size,
        random_state=random_state, stratify=y_trainval if stratify else None
    )

    logger.info("Split: train=%s, val=%s, test=%s",
                f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}")
    logger.info("Fraude: train=%.2f%%, val=%.2f%%, test=%.2f%%",
                y_train.mean()*100, y_val.mean()*100, y_test.mean()*100)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """Normaliza features com StandardScaler."""
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Converter y para arrays numpy
    y_train = y_train.values.astype(np.float32).reshape(-1, 1)
    y_val = y_val.values.astype(np.float32).reshape(-1, 1)
    y_test = y_test.values.astype(np.float32).reshape(-1, 1)

    logger.info("Features normalizadas: train=%s, val=%s, test=%s",
                X_train_scaled.shape, X_val_scaled.shape, X_test_scaled.shape)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=1024, patience=15):
    """Treina o modelo com early stopping baseado em F-beta."""
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_fbeta",
        mode="max",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_fbeta",
        mode="max",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
    
    fbeta_cb = FBetaCallback(validation_data=(X_val, y_val), beta=1.75)

    # Calcular class_weight para desbalanceamento
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    class_weight = {0: 1.0, 1: float(n_neg / n_pos)}
    logger.info("Class weights: %s", class_weight)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, fbeta_cb],
        class_weight=class_weight,
        verbose=1,
        shuffle=True
    )

    return model, history, fbeta_cb.best_fbeta


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Avalia o modelo no conjunto de teste."""
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_true = y_test.flatten()
    
    # Métricas com threshold padrão
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
    }
    
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()
    
    logger.info("Test (threshold=%.4f): ROC-AUC=%.4f, PR-AUC=%.4f, F1=%.4f, Prec=%.4f, Rec=%.4f",
                threshold, metrics["roc_auc"], metrics["pr_auc"], metrics["f1"],
                metrics["precision"], metrics["recall"])
    
    return metrics, y_pred_proba


def find_best_threshold(y_true, y_pred_proba, beta=1.75):
    """Encontra threshold ótimo para F-beta."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    fbeta_scores = (1 + beta**2) * (precisions * (recalls) / (beta**2 * precisions + recalls + 1e-10))
    best_idx = np.argmax(fbeta_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_fbeta = fbeta_scores[best_idx]
    return best_threshold, best_fbeta, precisions[best_idx], recalls[best_idx]


def optimize_threshold(model, X_val, y_val):
    """Otimiza threshold no conjunto de validação."""
    y_pred_proba = model.predict(X_val, verbose=0).flatten()
    y_true = y_val.flatten()
    
    best_threshold, best_fbeta, best_prec, best_recall = find_best_threshold(y_true, y_pred_proba, beta=2.0)
    logger.info("Best threshold: %.4f | F-beta: %.4f | Prec: %.4f | Rec: %.4f",
                best_threshold, best_fbeta, best_prec, best_recall)
    return best_threshold


def evaluate_with_threshold(model, X_test, y_test, threshold):
    """Avalia com threshold otimizado."""
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_true = y_test.flatten()
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
    }
    
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()
    
    logger.info("Test (threshold=%.4f): ROC-AUC=%.4f, PR-AUC=%.4f, F1=%.4f, Prec=%.4f, Rec=%.4f",
                threshold, metrics["roc_auc"], metrics["pr_auc"], metrics["f1"],
                metrics["precision"], metrics["recall"])
    
    return metrics, y_pred_proba


def json_serializable(obj):
    """Converte tipos numpy para tipos Python nativos para JSON."""
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


def save_results(model, scaler, test_metrics, best_threshold, history, best_params):
    """Salva modelo, scaler e métricas."""
    
    # Salvar modelo Keras (formato .keras nativo)
    model.save(MODEL_OUTPUT)
    logger.info("Modelo salvo: %s", MODEL_OUTPUT)
    
    # Salvar scaler
    with SCALER_OUTPUT.open("wb") as fh:
        pickle.dump(scaler, fh)
    logger.info("Scaler salvo: %s", SCALER_OUTPUT)
    
    # Histórico de treino
    history_dict = {}
    if history and hasattr(history, 'history'):
        for k, v in history.history.items():
            history_dict[k] = [float(x) for x in v]
    
    # Métricas completas
    full = {
        "model_name": "Keras_FraudDetectionNet",
        "best_params": best_params,
        "best_threshold": float(best_threshold),
        "balance_method": "class_weight",
        "test_metrics": test_metrics,
        "training_history": history_dict,
        "config": CFG,
    }
    
    full = json_serializable(full)
    
    with METRICS_OUTPUT.open("w") as fh:
        json.dump(full, fh, indent=2, ensure_ascii=False)
    logger.info("Métricas salvas: %s", METRICS_OUTPUT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Treino Deep Learning Fraude (Keras)")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs de treino")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--hidden-dims", type=str, default="512,256,128,64",
                        help="Dimensões das camadas ocultas (separadas por vírgula)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--l2-reg", type=float, default=1e-4, help="L2 regularization")
    args = parser.parse_args()

    # Parse hidden_dims
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",")]

    logger.info("=" * 60)
    logger.info("DEEP LEARNING (Keras) - Fraud Detection")
    logger.info("=" * 60)

    # Carregar dados
    df = load_abt()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Preparar dados
    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler = prepare_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # Modelo
    input_dim = X_train_scaled.shape[1]
    model = build_model(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        learning_rate=args.lr,
        l2_reg=args.l2_reg
    )
    
    logger.info("Modelo:\n%s", model.summary())
    logger.info("Parâmetros totais: %s", model.count_params())

    # Treino
    logger.info("=" * 60)
    logger.info("TREINAMENTO")
    logger.info("=" * 60)
    model, history, best_fbeta = train_model(
        model, X_train_scaled, y_train, X_val_scaled, y_val,
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience
    )

    # Otimizar threshold
    logger.info("Otimizando threshold...")
    best_threshold = optimize_threshold(model, X_val_scaled, y_val)

    # Avaliação final
    logger.info("=" * 60)
    logger.info("AVALIAÇÃO FINAL")
    logger.info("=" * 60)
    test_metrics, y_pred_proba = evaluate_model(model, X_test_scaled, y_test, threshold=0.5)

    # Avaliação com threshold otimizado
    opt_metrics, _ = evaluate_with_threshold(model, X_test_scaled, y_test, best_threshold)

    # Parâmetros para salvar
    best_params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "patience": args.patience,
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "l2_reg": args.l2_reg,
        "input_dim": input_dim,
    }

    # Salvar
    save_results(model, scaler, opt_metrics, best_threshold, history, best_params)

    print("\n" + "=" * 60)
    print("TREINAMENTO DL (Keras) CONCLUÍDO")
    print("=" * 60)
    print(f"Test ROC-AUC : {opt_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC  : {opt_metrics['pr_auc']:.4f}")
    print(f"Test F1      : {opt_metrics['f1']:.4f}")
    print(f"Test Precision: {opt_metrics['precision']:.4f}")
    print(f"Test Recall  : {opt_metrics['recall']:.4f}")
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best Val F-beta: {best_fbeta:.4f}")
    print(f"Model saved  : {MODEL_OUTPUT}")
    print(f"Scaler saved : {SCALER_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()