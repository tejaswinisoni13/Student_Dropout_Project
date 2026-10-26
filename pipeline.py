"""
Shared data-preprocessing and K-Means pipeline for the Student Performance
Analytics Dashboard. Supports 'mtech' and 'xapi' schemas automatically.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_CLUSTERS = 3

# Schema 1: Mtech-style academic dataset
MTECH_FEATURE_COLS = [
    "10th Score(%)",
    "12th Score(%)",
    "CGPA(%)",
    "AVG",
    "QUIZ Attempt",
    "Attendance",
    "Aim/Objective",
    "Prior Knowledge of subject",
    "Resource Visited",
    "No. of Replies Posted",
    "No. of views on forum",
]
QUIZ_COLS = ["QUIZ 1", "QUIZ 2", "QUIZ 3", "QUIZ 4", "QUIZ 5", "QUIZ 6"]

# Schema 2: xAPI-Edu-Data-style behavioral dataset
XAPI_NUMERIC_COLS = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
XAPI_BINARY_MAP = {
    "StudentAbsenceDays": {"Under-7": 0, "Above-7": 1},
    "ParentAnsweringSurvey": {"No": 0, "Yes": 1},
    "ParentschoolSatisfaction": {"Bad": 0, "Good": 1},
}
XAPI_FEATURE_COLS = XAPI_NUMERIC_COLS + list(XAPI_BINARY_MAP.keys())

SCORE_LABELS = {
    "mtech": "Avg Academic Score",
    "xapi": "Avg Engagement Score",
}

CLUSTER_COLORS = {
    "High-Performing": "#10b981",
    "Moderate": "#f59e0b",
    "At-Risk": "#ef4444",
}

RISK_LEVEL = {
    "High-Performing": "Low",
    "Moderate": "Medium",
    "At-Risk": "High",
}


def clean_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = df.columns.astype(str).str.strip()
    if "Roll Number" in df.columns and "Student ID" not in df.columns:
        df = df.rename(columns={"Roll Number": "Student ID"})
    return df


def get_id_column(df: pd.DataFrame):
    for candidate in ("Student ID", "Roll Number"):
        if candidate in df.columns:
            return candidate
    return None


def detect_dataset_type(df: pd.DataFrame):
    mtech_hits = sum(c in df.columns for c in MTECH_FEATURE_COLS)
    xapi_hits = sum(c in df.columns for c in XAPI_NUMERIC_COLS + list(XAPI_BINARY_MAP.keys()))

    if mtech_hits >= 3 and mtech_hits >= xapi_hits:
        return "mtech"
    if xapi_hits >= 3:
        return "xapi"
    return None


def validate_numeric_features(df: pd.DataFrame, feature_cols):
    df = df.copy()
    invalid_counts = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        before = df[col].isna().sum()
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid = int(converted.isna().sum() - before)
        if invalid > 0:
            invalid_counts[col] = invalid
        df[col] = converted
    return df, invalid_counts


def detect_iqr_outliers(df: pd.DataFrame, feature_cols):
    result = {}
    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            count = 0
        else:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            count = int(((s < lo) | (s > hi)).sum())
        if count:
            result[col] = count
    return result


def _standard_numeric_pipeline(df: pd.DataFrame, available_cols):
    df, invalid_counts = validate_numeric_features(df, available_cols)
    outliers = detect_iqr_outliers(df, available_cols)

    clipping_bounds = {}
    for col in available_cols:
        lo = float(df[col].quantile(0.01))
        hi = float(df[col].quantile(0.99))
        clipping_bounds[col] = (lo, hi)
        df[col] = df[col].clip(lower=lo, upper=hi)

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(df[available_cols])
    df[available_cols] = X_imputed

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    df["pca_x"] = X_pca[:, 0]
    df["pca_y"] = X_pca[:, 1]

    return {
        "df": df,
        "X_scaled": X_scaled,
        "available_cols": available_cols,
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "pca_variance": pca.explained_variance_ratio_,
        "invalid_counts": invalid_counts,
        "outliers": outliers,
        "clipping_bounds": clipping_bounds,
    }


def _prep_mtech(df: pd.DataFrame) -> dict:
    available_cols = [c for c in MTECH_FEATURE_COLS if c in df.columns]
    if not available_cols:
        raise ValueError("No Mtech-style feature columns found.")

    result = _standard_numeric_pipeline(df, available_cols)
    out_df = result["df"]
    out_df["primary_score"] = out_df["AVG"] if "AVG" in out_df.columns else out_df[available_cols].mean(axis=1)

    result["dataset_type"] = "mtech"
    result["quiz_cols"] = [c for c in QUIZ_COLS if c in out_df.columns]
    return result


def _prep_xapi(df: pd.DataFrame) -> dict:
    df = df.copy()
    if "Class" in df.columns:
        df["actual_class"] = df["Class"].map({"H": "High", "M": "Medium", "L": "Low"})

    for col, mapping in XAPI_BINARY_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    available_cols = [c for c in XAPI_FEATURE_COLS if c in df.columns]
    if not available_cols:
        raise ValueError("No xAPI-style feature columns found.")

    result = _standard_numeric_pipeline(df, available_cols)
    out_df = result["df"]
    numeric_present = [c for c in XAPI_NUMERIC_COLS if c in out_df.columns]
    out_df["primary_score"] = out_df[numeric_present].mean(axis=1) if numeric_present else out_df[available_cols].mean(axis=1)

    result["dataset_type"] = "xapi"
    result["quiz_cols"] = []
    return result


def preprocess(df_raw: pd.DataFrame, dataset_type: str = None) -> dict:
    df = clean_columns(df_raw)
    dtype = dataset_type or detect_dataset_type(df)
    if dtype is None:
        raise ValueError("This file doesn't match either supported dataset schema.")

    result = _prep_mtech(df) if dtype == "mtech" else _prep_xapi(df)
    id_col = get_id_column(result["df"])
    if id_col is None:
        result["df"].insert(0, "Student ID", range(1, len(result["df"]) + 1))
        id_col = "Student ID"

    result["id_col"] = id_col
    result["score_label"] = SCORE_LABELS.get(dtype, "Avg Score")
    return result


def evaluate_k_values(X_scaled, k_values=range(2, 7)) -> pd.DataFrame:
    rows = []
    n_samples = len(X_scaled)
    for k in k_values:
        if k < 2 or k >= n_samples:
            continue
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_scaled)
        rows.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette": silhouette_score(X_scaled, labels),
            "Davies-Bouldin": davies_bouldin_score(X_scaled, labels),
        })
    return pd.DataFrame(rows)


def _cluster_performance_scores(df: pd.DataFrame, label_col: str, feature_cols):
    normalized = df[feature_cols].copy()
    for col in feature_cols:
        lo, hi = normalized[col].min(), normalized[col].max()
        rng = hi - lo
        normalized[col] = (normalized[col] - lo) / rng if rng != 0 else 0.0

    return {
        cid: float(normalized.loc[df[label_col] == cid].mean().mean())
        for cid in sorted(df[label_col].unique())
    }


def assign_risk_tiers(df, label_col="kmeans_label", output_col="cluster", feature_cols=None):
    feature_cols = list(feature_cols or MTECH_FEATURE_COLS)
    n_clusters = len(df[label_col].unique())
    if n_clusters < 2:
        raise ValueError("Need at least 2 clusters to assign risk tiers.")

    scores = _cluster_performance_scores(df, label_col, feature_cols)
    ranked = sorted(scores, key=scores.get, reverse=True)

    mapping = {}
    for i, cid in enumerate(ranked):
        if i == 0:
            mapping[cid] = "High-Performing"
        elif i == n_clusters - 1:
            mapping[cid] = "At-Risk"
        else:
            mapping[cid] = "Moderate"

    result = df.copy()
    result[output_col] = result[label_col].map(mapping)
    return result, scores


def run_kmeans(prep: dict, n_clusters=N_CLUSTERS):
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    df = prep["df"].copy()
    X_scaled = prep["X_scaled"]
    feature_cols = prep["available_cols"]

    n_samples = len(df)
    if n_clusters >= n_samples:
        raise ValueError(f"n_clusters ({n_clusters}) must be smaller than number of students ({n_samples}).")

    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    df["kmeans_label"] = model.fit_predict(X_scaled)

    df, cluster_scores = assign_risk_tiers(df, label_col="kmeans_label", output_col="cluster", feature_cols=feature_cols)
    df["risk_level"] = df["cluster"].map(RISK_LEVEL)

    metrics = {
        "n_clusters": n_clusters,
        "silhouette": float(silhouette_score(X_scaled, df["kmeans_label"])),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, df["kmeans_label"])),
        "cluster_sizes": df["cluster"].value_counts().to_dict(),
        "cluster_performance_scores": cluster_scores,
    }

    if "actual_class" in df.columns:
        metrics["ari_vs_actual_class"] = float(adjusted_rand_score(df["actual_class"], df["kmeans_label"]))
        metrics["nmi_vs_actual_class"] = float(normalized_mutual_info_score(df["actual_class"], df["kmeans_label"]))

    return df, model, metrics


def add_derived_grade(df: pd.DataFrame, score_col="primary_score") -> pd.DataFrame:
    result = df.copy()
    if score_col not in result.columns:
        return result

    result["Grade"] = pd.cut(
        result[score_col],
        bins=[-np.inf, 50, 60, 75, np.inf],
        labels=["F", "B", "A", "O"],
        right=False,
    )
    return result
