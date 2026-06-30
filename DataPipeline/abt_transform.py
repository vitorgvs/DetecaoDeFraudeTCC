#!/usr/bin/env python
"""
abt_transform.py — Transformação dos dados limpos em ABT (Analytical Base Table).

Lê Dados/clean_data.csv, aplica feature engineering, encoding, seleção de features,
train/test split e normalização, exportando Dados/abt.csv.

Insights do EDA (exp_analysis.ipynb) incorporados:
- Missingness as signal: flags para colunas com >30% nulos
- TransactionAmt: log1p obrigatório (skewness 14)
- ProductCD: C (cashout) = 11.7% fraude → feature is_cashout
- card6: crédito 6.7% vs débito 2.4% → feature is_credit
- HourOfDay: extrair de TransactionDT, features cíclicas (sin/cos) + risk flags
- DeviceType: mobile 10.2%, desktop 6.5%, missing 2.1% → flags + encoding
- Variáveis V: top preditoras, priorizar no feature selection
- Variáveis M: flags de missing (NaN = verificação não possível = risco)
- Variáveis D: D1 correlação negativa (contas recentes = risco)
- Variáveis C: correlações lineares fracas mas não-lineares importantes
- Target Encoding com smoothing (alpha=10) para alta cardinalidade
- StandardScaler em todas features numéricas

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


def create_missing_flags(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Cria flags binárias _missing para colunas com 30-70% nulos (missingness as signal)."""
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
    """Aplica transformações numéricas baseadas nos insights do EDA."""
    # log1p(TransactionAmt) — EDA: skewness 14, cauda longa
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        skew_before = df["TransactionAmt"].skew()
        skew_after = df["TransactionAmt_log"].skew()
        logger.info(
            "TransactionAmt log1p: skewness %.2f -> %.2f",
            skew_before,
            skew_after,
        )

    # HourOfDay + features cíclicas — EDA: pico ~10% vs vale ~2.3% (4.5x diferença)
    if "TransactionDT" in df.columns:
        df["HourOfDay"] = (df["TransactionDT"] // 3600) % 24
        # Features cíclicas para capturar periodicidade
        df["HourOfDay_sin"] = np.sin(2 * np.pi * df["HourOfDay"] / 24)
        df["HourOfDay_cos"] = np.cos(2 * np.pi * df["HourOfDay"] / 24)
        # Flags de risco baseadas no EDA
        df["HourOfDay_risk_high"] = (
            ((df["HourOfDay"] >= 0) & (df["HourOfDay"] <= 6)).astype(np.int8)
        )  # madrugada/manhã cedo
        df["HourOfDay_risk_low"] = (
            ((df["HourOfDay"] >= 10) & (df["HourOfDay"] <= 18)).astype(np.int8)
        )  # horário comercial
        df["IsWeekend"] = ((df["HourOfDay"] // 24) % 7 >= 5).astype(np.int8)
        logger.info("HourOfDay + features cíclicas + risk flags criadas")

    # ProductCD: cashout (C) = 11.7% fraude — EDA hotspot crítico
    if "ProductCD" in df.columns:
        df["ProductCD_is_cashout"] = (df["ProductCD"] == "C").astype(np.int8)
        df["ProductCD_encoded"] = df["ProductCD"].astype("category").cat.codes
        df["ProductCD_freq"] = df.groupby("ProductCD")["ProductCD"].transform("count")
        logger.info("ProductCD features: is_cashout, encoded, freq")

    # card6: crédito vs débito — EDA: 6.7% vs 2.4%
    if "card6" in df.columns:
        df["card6_is_credit"] = (df["card6"] == "credit").astype(np.int8)
        df["card6_encoded"] = df["card6"].astype("category").cat.codes
        logger.info("card6 features: is_credit, encoded")

    # card4: Discover tem 7.7% fraude (1.1% volume) — EDA
    if "card4" in df.columns:
        df["card4_is_discover"] = (df["card4"] == "discover").astype(np.int8)
        df["card4_encoded"] = df["card4"].astype("category").cat.codes
        logger.info("card4 features: is_discover, encoded")

    # DeviceType: mobile 10.2%, desktop 6.5%, missing 2.1% — EDA
    if "DeviceType" in df.columns:
        df["DeviceType_is_mobile"] = (df["DeviceType"] == "mobile").astype(np.int8)
        df["DeviceType_is_desktop"] = (df["DeviceType"] == "desktop").astype(np.int8)
        df["DeviceType_missing"] = df["DeviceType"].isnull().astype(np.int8)
        df["DeviceType_encoded"] = (
            df["DeviceType"].fillna("missing").astype("category").cat.codes
        )
        logger.info("DeviceType features: is_mobile, is_desktop, missing, encoded")

    # DeviceInfo: frequência + rare
    if "DeviceInfo" in df.columns:
        device_freq = df["DeviceInfo"].value_counts()
        df["DeviceInfo_freq"] = df["DeviceInfo"].map(device_freq).fillna(0).astype(int)
        df["DeviceInfo_rare"] = (df["DeviceInfo_freq"] < 10).astype(np.int8)
        logger.info("DeviceInfo features: freq, rare")

    # Aggregações C (Contagem) — correlações lineares fracas mas não-lineares importantes
    c_cols = [c for c in df.columns if c[0] == "C" and c[1:].isdigit()]
    if c_cols:
        df["C_sum"] = df[c_cols].sum(axis=1)
        df["C_mean"] = df[c_cols].mean(axis=1)
        df["C_max"] = df[c_cols].max(axis=1)
        df["C_nonzero_count"] = (df[c_cols] != 0).sum(axis=1)
        df["C_all_zero"] = (df[c_cols].sum(axis=1) == 0).astype(np.int8)
        logger.info("C-variables aggregations: sum, mean, max, nonzero_count, all_zero")

    # Aggregações D (Temporal) — D1 correlação negativa com fraude
    d_cols = [c for c in df.columns if c[0] == "D" and c[1:].isdigit()]
    if d_cols and "D1" in df.columns:
        df["D1_recent"] = (df["D1"] <= 1).astype(np.int8)  # conta muito recente
        df["D1_very_recent"] = (df["D1"] == 0).astype(np.int8)
        df["D1_missing"] = df["D1"].isnull().astype(np.int8)
        df["D_mean"] = df[d_cols].mean(axis=1)
        df["D_min"] = df[d_cols].min(axis=1)
        df["D_max"] = df[d_cols].max(axis=1)
        df["D_missing_count"] = df[d_cols].isnull().sum(axis=1)
        df["D1_log"] = np.log1p(df["D1"].fillna(0))
        logger.info("D-variables aggregations + D1 specific features")

    # Aggregações M (Match) — flags de verificação
    m_cols = [c for c in df.columns if c[0] == "M" and c[1:].isdigit()]
    if m_cols:
        df["M_failure_count"] = (df[m_cols] == "F").sum(axis=1)
        df["M_missing_count"] = df[m_cols].isnull().sum(axis=1)
        df["M_success_count"] = (df[m_cols] == "T").sum(axis=1)
        df["M_any_failure"] = (df[m_cols] == "F").any(axis=1).astype(np.int8)
        logger.info("M-variables aggregations: failure_count, missing_count, success_count, any_failure")

    # Identidade: presence + missing count
    id_cols = [c for c in df.columns if c.startswith("id_")]
    if id_cols:
        df["has_identity"] = df[id_cols].notnull().any(axis=1).astype(np.int8)
        df["identity_missing_count"] = df[id_cols].isnull().sum(axis=1)
        df["identity_present_count"] = df[id_cols].notnull().sum(axis=1)
        df["device_completely_missing"] = (
            df["DeviceType"].isnull() & df["DeviceInfo"].isnull()
        ).astype(np.int8)
        logger.info("Identity features: has_identity, missing_count, present_count, device_completely_missing")

    # V-variables: top 20 sum/mean (mais preditivas segundo EDA)
    v_cols = [c for c in df.columns if c[0] == "V" and c[1:].isdigit()]
    if v_cols:
        # Selecionar top 20 mais correlacionadas (pre-calculado no EDA)
        top_v = [
            "V258", "V294", "V91", "V70", "V201", "V172", "V187", "V161", "V226",
            "V324", "V177", "V189", "V108", "V110", "V259", "V284", "V283", "V271",
            "V243", "V260"
        ]
        top_v_exist = [c for c in top_v if c in df.columns]
        if top_v_exist:
            df["V_top20_sum"] = df[top_v_exist].sum(axis=1)
            df["V_top20_mean"] = df[top_v_exist].mean(axis=1)
            logger.info("V_top20 features: sum, mean (from %s vars)", len(top_v_exist))

    # Interaction features baseadas no EDA
    if "TransactionAmt" in df.columns and "ProductCD" in df.columns:
        df["Amt_x_cashout"] = df["TransactionAmt"] * df["ProductCD_is_cashout"]
    if "HourOfDay" in df.columns and "DeviceType" in df.columns:
        df["Night_x_mobile"] = df["HourOfDay_risk_high"] * df["DeviceType_is_mobile"]
    if "TransactionAmt" in df.columns and "card6" in df.columns:
        df["Amt_x_credit"] = df["TransactionAmt"] * df["card6_is_credit"]
    if "has_identity" in df.columns and "TransactionAmt" in df.columns:
        df["no_id_x_high_amt"] = (
            (df["has_identity"] == 0) & (df["TransactionAmt"] > df["TransactionAmt"].quantile(0.9))
        ).astype(np.int8)
    if "has_identity" in df.columns and "M_any_failure" in df.columns:
        df["no_id_x_M_fail"] = (
            (df["has_identity"] == 0) & (df["M_any_failure"] == 1)
        ).astype(np.int8)

    # Completeness score e risk flag sum
    missing_flag_cols = [c for c in df.columns if c.endswith("_missing")]
    if missing_flag_cols:
        df["completeness_score"] = 1 - (df[missing_flag_cols].sum(axis=1) / len(missing_flag_cols))
        df["risk_flag_sum"] = df[missing_flag_cols].sum(axis=1)

    # Imputar NaNs numéricos com mediana (exceto target e originais)
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
    """Aplica Label Encoding nas colunas categóricas configuradas + auto-detect."""
    encoders = {}
    mappings = {}

    # Colunas configuradas para Label Encoding
    for col in ABT_CFG["label_encode_cols"]:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("__MISSING__").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        mappings[col] = len(le.classes_)
        logger.info("LabelEncoded %s (%s categorias)", col, mappings[col])

    # Quaisquer categorias restantes (object dtype)
    remaining = df.select_dtypes(include=["object"]).columns.tolist()
    if remaining:
        logger.warning(
            "Categorias sem encoding configurado (auto-encoding): %s", remaining
        )
        for col in remaining:
            df[col] = df[col].fillna("__MISSING__").astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            mappings[col] = len(le.classes_)
            logger.info("LabelEncoded (auto) %s (%s categorias)", col, mappings[col])

    return df, mappings


def select_features(df: pd.DataFrame, missing_flags: list[str]) -> list[str]:
    """Seleciona features para a ABT baseada na importância do EDA."""
    miss_pct = df.isnull().mean() * 100
    cols_high_missing = miss_pct[miss_pct > ABT_CFG["drop_cols_above_pct"]].index.tolist()
    cols_high_missing = [
        c for c in cols_high_missing if c != "isFraud" and c in df.columns
    ]

    # Grupos de features priorizados pelo EDA
    v_cols = [c for c in df.columns if c[0] == "V" and c[1:].isdigit()]
    c_cols = [c for c in df.columns if c[0] == "C" and c[1:].isdigit()]
    d_cols = [c for c in df.columns if c[0] == "D" and c[1:].isdigit()]
    m_cols = [c for c in df.columns if c[0] == "M" and c[1:].isdigit()]
    card_cols = [c for c in df.columns if c.startswith("card")]
    addr_cols = [c for c in df.columns if c.startswith("addr") or c.startswith("dist")]
    email_cols = [c for c in df.columns if c.endswith("emaildomain")]

    # Features engineered (prioritárias segundo EDA)
    engineered_cols = [
        c
        for c in [
            "ProductCD_is_cashout",
            "ProductCD_encoded",
            "ProductCD_freq",
            "card6_is_credit",
            "card6_encoded",
            "card4_is_discover",
            "card4_encoded",
            "card1_freq",
            "card2_freq",
            "card3_freq",
            "HourOfDay_sin",
            "HourOfDay_cos",
            "HourOfDay_risk_high",
            "HourOfDay_risk_low",
            "IsWeekend",
            "DeviceType_is_mobile",
            "DeviceType_is_desktop",
            "DeviceType_missing",
            "DeviceType_encoded",
            "DeviceInfo_freq",
            "DeviceInfo_rare",
            "TransactionAmt_log",
            "C_sum",
            "C_mean",
            "C_max",
            "C_nonzero_count",
            "C_all_zero",
            "D1_recent",
            "D1_very_recent",
            "D1_missing",
            "D_mean",
            "D_min",
            "D_max",
            "D_missing_count",
            "D1_log",
            "M_failure_count",
            "M_missing_count",
            "M_success_count",
            "M_any_failure",
            "has_identity",
            "identity_missing_count",
            "identity_present_count",
            "device_completely_missing",
            "V_top20_sum",
            "V_top20_mean",
            "Amt_x_cashout",
            "Night_x_mobile",
            "Amt_x_credit",
            "no_id_x_high_amt",
            "no_id_x_M_fail",
            "completeness_score",
            "risk_flag_sum",
        ]
        if c in df.columns
    ]

    # Missing flags específicas do EDA (D, M, V, etc)
    specific_missing_flags = [
        c for c in missing_flags
        if c.startswith(("D", "M", "V", "P_emaildomain"))
    ]

    selected = (
        v_cols
        + c_cols
        + d_cols
        + m_cols
        + card_cols
        + addr_cols
        + email_cols
        + engineered_cols
        + specific_missing_flags
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


def train_test_split_abt(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    # 3. Missing flags (missingness as signal — EDA insight)
    df, missing_flags = create_missing_flags(df)

    # 4. Transformações (feature engineering baseada no EDA)
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
        "engineered_features": len([c for c in selected_features if c in [
            "ProductCD_is_cashout", "ProductCD_encoded", "ProductCD_freq",
            "card6_is_credit", "card6_encoded", "card4_is_discover", "card4_encoded",
            "card1_freq", "card2_freq", "card3_freq",
            "HourOfDay_sin", "HourOfDay_cos", "HourOfDay_risk_high", "HourOfDay_risk_low",
            "IsWeekend", "DeviceType_is_mobile", "DeviceType_is_desktop", "DeviceType_missing",
            "DeviceType_encoded", "DeviceInfo_freq", "DeviceInfo_rare",
            "TransactionAmt_log", "C_sum", "C_mean", "C_max", "C_nonzero_count", "C_all_zero",
            "D1_recent", "D1_very_recent", "D1_missing", "D_mean", "D_min", "D_max",
            "D_missing_count", "D1_log", "M_failure_count", "M_missing_count",
            "M_success_count", "M_any_failure", "has_identity", "identity_missing_count",
            "identity_present_count", "device_completely_missing", "V_top20_sum", "V_top20_mean",
            "Amt_x_cashout", "Night_x_mobile", "Amt_x_credit", "no_id_x_high_amt",
            "no_id_x_M_fail", "completeness_score", "risk_flag_sum"
        ]]),
        "missing_flags": len([c for c in selected_features if c.endswith("_missing")]),
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
            "HourOfDay extracted from TransactionDT + sin/cos cyclical + risk flags",
            "ProductCD: is_cashout (C=11.7% fraud), encoded, freq",
            "card6: is_credit (credit=6.7% vs debit=2.4%), encoded",
            "card4: is_discover (7.7% fraud), encoded",
            "DeviceType: is_mobile (10.2%), is_desktop (6.5%), missing (2.1%), encoded",
            "DeviceInfo: freq, rare",
            "C-variables: sum, mean, max, nonzero_count, all_zero",
            "D-variables: D1_recent/very_recent/missing, mean/min/max/missing_count, D1_log",
            "M-variables: failure_count, missing_count, success_count, any_failure",
            "Identity: has_identity, missing/present_count, device_completely_missing",
            "V-variables: V_top20_sum, V_top20_mean (top 20 from EDA)",
            "Interactions: Amt_x_cashout, Night_x_mobile, Amt_x_credit, no_id_x_high_amt, no_id_x_M_fail",
            "Composite: completeness_score, risk_flag_sum",
            f"Label Encoding: {', '.join(encoding_mappings.keys())}",
            "StandardScaler applied to all numeric features",
            f"Train/Test split (test_size={ABT_CFG['test_size']}, stratified={ABT_CFG['stratify_target']})",
            "Missingness as signal: flags for 30-70% missing columns",
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
    print("Feature Groups:")
    for k, v in feature_groups.items():
        print(f"  {k}: {v}")
    print()
    print("Top 10 preditores:")
    for c in correlations[:10]:
        sign = "+" if c["correlation"] > 0 else "-"
        print(f"  {c['feature']:<30} {sign}{abs(c['correlation']):.4f}")
    print()
    print(f"ABT            : {abt_path}")
    print(f"Metadata       : {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()