from __future__ import annotations

from itertools import permutations
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from preprocess import get_feature_columns


SEGMENT_ARCHETYPES = {
    "profil_academique": [
        "implication_academique",
        "interet_contenu_pedagogique",
    ],
    "profil_associatif_evenementiel": [
        "implication_vie_etudiante",
        "interet_evenementiel",
    ],
    "profil_technique_vigilant": [
        "niveau_technique_estime",
        "aisance_numerique",
    ],
}


def _compute_segment_scores(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    """Calcule un score d'affinité entre chaque cluster et chaque segment métier."""
    segment_scores = {
        segment_name: cluster_summary[columns].mean(axis=1)
        for segment_name, columns in SEGMENT_ARCHETYPES.items()
    }

    hybrid_balance = (
        3
        - (
            cluster_summary["implication_academique"]
            - cluster_summary["implication_vie_etudiante"]
        ).abs()
    )
    segment_scores["profil_hybride_social"] = pd.concat(
        [
            cluster_summary[
                [
                    "implication_vie_etudiante",
                    "reactivite_percue",
                    "presence_numerique_visible",
                ]
            ],
            hybrid_balance.rename("equilibre_hybride"),
        ],
        axis=1,
    ).mean(axis=1)

    return pd.DataFrame(segment_scores)


def _assign_segment_labels(cluster_summary: pd.DataFrame) -> dict[int, str]:
    """Associe un libellé métier unique à chaque cluster."""
    segment_scores = _compute_segment_scores(cluster_summary)
    cluster_ids = segment_scores.index.tolist()
    segment_names = segment_scores.columns.tolist()

    best_mapping: dict[int, str] = {}
    best_total_score = -np.inf

    for candidate_labels in permutations(segment_names, len(cluster_ids)):
        total_score = sum(
            segment_scores.loc[cluster_id, segment_name]
            for cluster_id, segment_name in zip(cluster_ids, candidate_labels)
        )
        if total_score > best_total_score:
            best_total_score = total_score
            best_mapping = dict(zip(cluster_ids, candidate_labels))

    return best_mapping


def _compute_confidence_scores(model: KMeans, x_scaled: np.ndarray) -> np.ndarray:
    """Produit un score de confiance pseudo-probabiliste à partir des distances KMeans."""
    distances = model.transform(x_scaled)
    inverse_distances = 1 / (1 + distances)
    confidence_scores = inverse_distances.max(axis=1) / inverse_distances.sum(axis=1)
    return np.round(confidence_scores, 3)


def cluster_profiles(
    df: pd.DataFrame,
    n_clusters: int = 4,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Segmente les profils avec KMeans et ajoute un segment métier lisible."""
    feature_cols = get_feature_columns()
    x = df[feature_cols]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(x_scaled)

    result = df.copy()
    result["cluster"] = clusters
    cluster_summary = result.groupby("cluster")[feature_cols].mean()
    segment_mapping = _assign_segment_labels(cluster_summary)
    result["segment_principal"] = result["cluster"].map(segment_mapping)
    result["score_confiance_segment"] = _compute_confidence_scores(model, x_scaled)

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

    summary = df.groupby("cluster")[numeric_cols].mean().round(2)

    if "segment_principal" in df.columns:
        segment_summary = df.groupby("cluster")["segment_principal"].first()
        profile_counts = df.groupby("cluster").size().rename("nb_profils")
        avg_confidence = (
            df.groupby("cluster")["score_confiance_segment"]
            .mean()
            .round(3)
            .rename("score_confiance_moyen")
        )
        summary = pd.concat(
            [segment_summary, profile_counts, avg_confidence, summary],
            axis=1,
        )

    return summary
