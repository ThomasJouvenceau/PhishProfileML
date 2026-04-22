from __future__ import annotations

import pandas as pd


STATUT_MAP = {
    "initial": 0,
    "alternance": 1,
}

SPECIALITE_MAP = {
    "cyber": 0,
    "dev": 1,
    "data": 2,
    "mixte": 3,
}

STYLE_MAP = {
    "formel": 0,
    "mixte": 1,
    "informel": 2,
}


def load_profiles(csv_path: str) -> pd.DataFrame:
    """Charge un fichier CSV de profils."""
    return pd.read_csv(csv_path)


def encode_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les variables catégorielles du dataset."""
    encoded = df.copy()

    encoded["statut"] = encoded["statut"].map(STATUT_MAP)
    encoded["specialite_dominante"] = encoded["specialite_dominante"].map(SPECIALITE_MAP)
    encoded["style_communication"] = encoded["style_communication"].map(STYLE_MAP)

    if encoded.isnull().sum().sum() > 0:
        missing_cols = encoded.columns[encoded.isnull().any()].tolist()
        raise ValueError(f"Valeurs manquantes après encodage dans : {missing_cols}")

    return encoded


def get_feature_columns() -> list[str]:
    """Retourne les colonnes utilisées pour le modèle."""
    return [
        "statut",
        "specialite_dominante",
        "implication_academique",
        "interet_contenu_pedagogique",
        "implication_vie_etudiante",
        "interet_evenementiel",
        "niveau_technique_estime",
        "aisance_numerique",
        "style_communication",
        "reactivite_percue",
        "presence_numerique_visible",
    ]