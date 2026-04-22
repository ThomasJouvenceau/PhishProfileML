from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from preprocess import get_feature_columns


def cluster_profiles(
    df: pd.DataFrame,
    n_clusters: int = 4,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Segmente les profils avec KMeans."""
    feature_cols = get_feature_columns()
    x = df[feature_cols]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(x_scaled)

    result = df.copy()
    result["cluster"] = clusters

    return result, model, scaler


def save_cluster_artifacts(
    model: KMeans,
    scaler: StandardScaler,
    output_dir: str = "models",
) -> None:
    """Sauvegarde le modèle et le scaler."""
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, "kmeans_profiles.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne un résumé moyen des clusters."""
    numeric_cols = [
        "implication_academique",
        "interet_contenu_pedagogique",
        "implication_vie_etudiante",
        "interet_evenementiel",
        "niveau_technique_estime",
        "aisance_numerique",
        "reactivite_percue",
        "presence_numerique_visible",
    ]

    return df.groupby("cluster")[numeric_cols].mean().round(2)