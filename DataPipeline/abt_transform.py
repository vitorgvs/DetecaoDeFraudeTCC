#!/usr/bin/env python
"""
abt_transform.py — Transformação dos dados limpos em ABT (Analytical Base Table).

Lê Dados/clean_data.csv, aplica feature engineering, encoding, seleção de features,
train/test split e normalização, exportando Dados/abt.csv.

Uso:
    python DataPipeline/abt_transform.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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
CONFIG_PATH = Path(__file__).with_name("pipeline_config.json")

with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    CFG = json.load(fh)

PATHS = CFG["paths"]
ABT_CFG = CFG["abt"]
META = CFG["metadata"]


# ---------------------------------------------------------------------------
# Funções utilitárias
# ---------------------------------------------------------------------------
def profile_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Classifica colunas em grupos lógicos."""
    groups = {
        "Target": ["isFraud"],
        "Transacao_base": [
            c for c in ["TransactionDT", "TransactionAmt", "ProductCD"] if c in df.columns
        ],
        "Cartao": [c for c in df.columns if c.startswith("card")],
        "Endereco_Dist": [
            c for c in df.columns if c.startswith("addr") or c.startswith("dist")
        ],
        "Email": [
            c for c in ["P_emaildomain", "R_emaildomain"] if c in df.columns
        ],
        "Contagem_C": [c for c in df.columns if c[0] == "C" and c[1:].isdigit()],
        "Temporal_D": [c for c in df.columns if c[0] == "D" and c[1:].isdigit()],
        "Match_M": [c for c in df.columns if c[0] == "M" and c[1:].isdigit()],
        "Vesta_V": [c for c in df.columns if c[0] == "V" and c[1:].isdigit()],
        "Identidade_id": [c for c in df.columns if c.startswith("id_")],
        "Dispositivo": [c for c in ["DeviceType", "DeviceInfo"] if c in df.columns],
    }
    return {k: v for k, v in groups.items() if v}


def create_missing_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Cria flags binárias _missing para colunas com 30-70% nulos."""
    miss_pct = df.isnull().mean() * 100
    structural_cols = miss_pct[
        (miss_pct > ABT_CFG["missing_flag_lower_pct"])
        & (miss_pct <= ABT_CFG["missing_flag_upper_pct"])
    ].index.tolist()

    flags = []
    for col in structural_cols:
        flag_name = f"{col}_missing"
        df[flag_name] = df[col].isnull().astype(np.int8)
        flags.append(flag_name)

    logger.info("Flags de missingness criadas: %s", len(flags))
    return df, flags


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformações numéricas: log1p, HourOfDay, imputação mediana."""
    # log1p(TransactionAmt)
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        skew_before = df["TransactionAmt"].skew()
        skew_after = df["TransactionAmt_log"].skew()
        logger.info(
            "TransactionAmt log1p: skewness %.2f -> %.2f",
            skew_before,
            skew_after,
        )

    # HourOfDay
    if "TransactionDT" in df.columns:
        df["HourOfDay"] = (df["TransactionDT"] // 3600) % 24
        logger.info("HourOfDay criada")

    # Imputar NaNs numéricos com mediana
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_impute = [
        c
        for c in numeric_cols
        if c not in ("isFraud", "TransactionDT", "TransactionAmt")
        and df[c].isnull().sum() > 0
    ]
    for col in cols_to_impute:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    logger.info("Imputação mediana: %s colunas", len(cols_to_impute))

    # Remover originais transformadas
    drop_cols = [c for c in ["TransactionDT", "TransactionAmt"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        logger.info("Removidas colunas originais: %s", drop_cols)

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aplica Label Encoding nas colunas categóricas."""
    encoders = {}
    mappings = {}

    # Colunas configuradas
    for col in ABT_CFG["label_encode_cols"]:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("__MISSING__").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        mappings[col] = len(le.classes_)
        logger.info("LabelEncoded %s (%s categorias)", col, mappings[col])

    # Quaisquer categorias restantes
    remaining = df.select_dtypes(include=["object"]).columns.tolist()
    if remaining:
        logger.warning("Categorias sem encoding configurado (auto-encoding): %s", remaining)
        for col in remaining:
            df[col] = df[col].fillna("__MISSING__").astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            mappings[col] = len(le.classes_)
            logger.info("LabelEncoded (auto) %s (%s categorias)", col, mappings[col])

    return df, mappings


def select_features(df: pd.DataFrame, missing_flags: list[str]) -> list[str]:
    """Seleciona features para a ABT."""
    miss_pct = df.isnull().mean() * 100
    cols_high_missing = miss_pct[miss_pct > ABT_CFG["drop_cols_above_pct"]].index.tolist()
    cols_high_missing = [c for c in cols_high_missing if c != "isFraud" and c in df.columns]

    # Grupos de features
    v_cols = [c for c in df.columns if c[0] == "V" and c[1:].isdigit()]
    c_cols = [c for c in df.columns if c[0] == "C" and c[1:].isdigit()]
    d_cols = [c for c in df.columns if c[0] == "D" and c[1:].isdigit()]
    m_cols = [c for c in df.columns if c[0] == "M" and c[1:].isdigit()]
    card_cols = [c for c in df.columns if c.startswith("card")]
    addr_cols = [c for c in df.columns if c.startswith("addr") or c.startswith("dist")]
    email_cols = [c for c in df.columns if c.endswith("emaildomain")]

    selected = (
        v_cols
        + c_cols
        + d_cols
        + m_cols
        + card_cols
        + [c for c in ["ProductCD", "HourOfDay", "TransactionAmt_log", "DeviceType"] if c in df.columns]
        + addr_cols
        + email_cols
        + missing_flags
        + ["isFraud"]
    )

    # Remover duplicatas e inexistentes
    selected = list(dict.fromkeys(c for c in selected if c in df.columns))
    # Excluir >70% nulos
    selected = [c for c in selected if c not in cols_high_missing]

    logger.info(
        "Features selecionadas: %s | Excluidas (>70%% nulos): %s",
        len(selected),
        len(cols_high_missing),
    )

    return selected


def impute_remaining_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Imputação final de qualquer NaN restante."""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col == "isFraud":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("__MISSING__")
                if df[col].dtype == object:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))

    still_nan = int(df.isnull().sum().sum())
    if still_nan > 0:
        logger.warning("Ainda existem %s NaN apos imputacao final", still_nan)
    else:
        logger.info("ABT sem NaN apos imputacao final")
    return df


def train_test_split_abt(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Realiza train/test split estratificado."""
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=ABT_CFG["test_size"],
        random_state=ABT_CFG["random_state"],
        stratify=y if ABT_CFG["stratify_target"] else None,
    )

    abt_train = X_train.copy()
    abt_train["isFraud"] = y_train.values

    abt_test = X_test.copy()
    abt_test["isFraud"] = y_test.values

    logger.info(
        "Train/Test split: train=%s, test=%s",
        f"{len(abt_train):,}",
        f"{len(abt_test):,}",
    )
    logger.info(
        "Fraude treino: %.2f%% | Fraude teste: %.2f%%",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )

    return abt_train, abt_test


def normalize_features(
    abt_train: pd.DataFrame, abt_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Aplica StandardScaler nas features numéricas."""
    if not ABT_CFG["apply_standard_scaler"]:
        logger.info("Normalização desabilitada")
        return abt_train, abt_test, StandardScaler()

    feature_cols = [c for c in abt_train.columns if c != "isFraud"]

    scaler = StandardScaler()
    train_scaled = abt_train.copy()
    train_scaled[feature_cols] = scaler.fit_transform(abt_train[feature_cols])

    test_scaled = abt_test.copy()
    test_scaled[feature_cols] = scaler.transform(abt_test[feature_cols])

    logger.info(
        "Normalização: media=%.4f, std=%.4f",
        train_scaled[feature_cols].mean().mean(),
        train_scaled[feature_cols].std().mean(),
    )

    return train_scaled, test_scaled, scaler


def compute_correlations(abt: pd.DataFrame) -> list[dict]:
    """Calcula correlações com o target."""
    feature_cols = [c for c in abt.columns if c != "isFraud"]
    corr_list = []

    for col in feature_cols:
        if abt[col].notna().sum() > 0:
            corr = abt[col].corr(abt["isFraud"])
            if not np.isnan(corr):
                corr_list.append({"feature": col, "correlation": round(float(corr), 6)})

    corr_list.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return corr_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=" * 60)
    logger.info("ABT TRANSFORM - Fraud Detection Pipeline")
    logger.info("=" * 60)

    # 1. Carregar dados limpos
    clean_path = Path(PATHS["clean_data"])
    logger.info("Carregando dados limpos: %s", clean_path)
    df = pd.read_csv(clean_path)

    # 2. Profiling
    groups = profile_columns(df)
    for name, cols in groups.items():
        logger.info("Grupo %-20s: %s variaveis", name, len(cols))

    # 3. Missing flags
    df, missing_flags = create_missing_flags(df)

    # 4. Transformações
    df = transform_features(df)

    # 5. Encoding
    df, encoding_mappings = encode_categoricals(df)

    # 6. Seleção de features
    selected_features = select_features(df, missing_flags)
    df = df[selected_features].copy()

    # 7. Imputação final
    df = impute_remaining_nans(df)

    # 8. Train/Test split
    abt_train, abt_test = train_test_split_abt(df)

    # 9. Normalização
    abt_train, abt_test, scaler = normalize_features(abt_train, abt_test)

    # 10. ABT completa (treino + teste normalizados)
    abt_full = pd.concat([abt_train, abt_test], axis=0).reset_index(drop=True)

    # 11. Correlações
    correlations = compute_correlations(abt_full)

    # 12. Feature groups metadata
    feature_groups = {
        "V_variables": len([c for c in selected_features if c[0] == "V" and c[1:].isdigit()]),
        "C_variables": len([c for c in selected_features if c[0] == "C" and c[1:].isdigit()]),
        "D_variables": len([c for c in selected_features if c[0] == "D" and c[1:].isdigit()]),
        "M_variables": len([c for c in selected_features if c[0] == "M" and c[1:].isdigit()]),
        "card_variables": len([c for c in selected_features if c.startswith("card")]),
        "missing_flags": len(missing_flags),
    }

    # 13. Exportar ABT
    abt_path = Path(PATHS["abt_data"])
    logger.info("Exportando ABT: %s", abt_path)
    abt_full.to_csv(abt_path, index=False)

    # 14. Metadata
    metadata = {
        "dataset_info": {
            "total_rows": int(len(abt_full)),
            "total_features": int(abt_full.shape[1] - 1),
            "target": META["target_column"],
            "train_rows": int(len(abt_train)),
            "test_rows": int(len(abt_test)),
            "fraud_rate": round(float(abt_full["isFraud"].mean()), 6),
        },
        "feature_groups": feature_groups,
        "transformations_applied": [
            "log1p(TransactionAmt) -> TransactionAmt_log",
            "HourOfDay extracted from TransactionDT",
            f"Label Encoding: {', '.join(encoding_mappings.keys())}",
            "StandardScaler applied to all numeric features",
            f"Train/Test split (test_size={ABT_CFG['test_size']}, stratified={ABT_CFG['stratify_target']})",
        ],
        "features_list": [c for c in abt_full.columns if c != "isFraud"],
        "strong_predictors": correlations[:20],
        "encoding_mappings": {k: int(v) for k, v in encoding_mappings.items()},
    }

    metadata_path = abt_path.with_name("abt_metadata.json")
    logger.info("Exportando metadata: %s", metadata_path)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    # Print summary
    print()
    print("=" * 60)
    print("ABT TRANSFORM CONCLUÍDA")
    print("=" * 60)
    print(f"Total de linhas : {len(abt_full):,}")
    print(f"Total de features: {abt_full.shape[1] - 1}")
    print(f"Treino          : {len(abt_train):,}")
    print(f"Teste           : {len(abt_test):,}")
    print(f"Taxa de fraude  : {abt_full['isFraud'].mean()*100:.2f}%")
    print()
    print("Top 5 preditores:")
    for c in correlations[:5]:
        sign = "+" if c["correlation"] > 0 else "-"
        print(f"  {c['feature']:<25} {sign}{abs(c['correlation']):.4f}")
    print()
    print(f"ABT            : {abt_path}")
    print(f"Metadata       : {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()