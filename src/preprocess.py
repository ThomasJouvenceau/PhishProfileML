from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROFILE_COLUMNS = [
    "id_profil",
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

LIKERT_COLUMNS = [
    "implication_academique",
    "interet_contenu_pedagogique",
    "implication_vie_etudiante",
    "interet_evenementiel",
    "niveau_technique_estime",
    "aisance_numerique",
    "reactivite_percue",
    "presence_numerique_visible",
]

DEFAULT_INPUT_CANDIDATES = (
    "data/raw_profiles.csv",
    "data/profiles_template.csv",
)
DEFAULT_OUTPUT_CSV = "data/processed_profiles.csv"

STATUT_MAP = {
    "initial": 0,
    "alternance": 1,
    "0": 0,
    "1": 1,
}

SPECIALITE_MAP = {
    "cyber": 0,
    "dev": 1,
    "data": 2,
    "mixte": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
}

STYLE_MAP = {
    "formel": 0,
    "mixte": 1,
    "informel": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}

LIKERT_MAP = {
    "faible": 1,
    "1": 1,
    "moyen": 2,
    "moyenne": 2,
    "2": 2,
    "fort": 3,
    "forte": 3,
    "3": 3,
}


def _normalize_token(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_profile_id(value: object) -> str:
    normalized = _normalize_token(value).replace(" ", "")

    if not normalized:
        return ""

    match = re.fullmatch(r"p?0*(\d+)", normalized)
    if match:
        return f"P{int(match.group(1)):02d}"

    return normalized.upper()


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in PROFILE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le dataset : "
            + ", ".join(missing_columns)
        )


def _encode_series(
    series: pd.Series,
    mapping: dict[str, int],
    column_name: str,
) -> pd.Series:
    normalized = series.map(_normalize_token)
    encoded = normalized.map(mapping)

    if encoded.isna().any():
        invalid_values = sorted(set(normalized[encoded.isna()]))
        raise ValueError(
            f"Valeurs invalides dans la colonne '{column_name}' : "
            + ", ".join(invalid_values)
        )

    return encoded.astype(int)


def load_profiles(csv_path: str) -> pd.DataFrame:
    """Charge un fichier CSV de profils."""
    return pd.read_csv(csv_path)


def resolve_input_csv(preferred_path: str | None = None) -> str:
    """Retourne le premier fichier source disponible."""
    candidate_paths = (
        [preferred_path]
        if preferred_path is not None
        else list(DEFAULT_INPUT_CANDIDATES)
    )

    for candidate in candidate_paths:
        if candidate and Path(candidate).exists():
            return candidate

    searched_paths = ", ".join(path for path in candidate_paths if path)
    raise FileNotFoundError(
        "Aucun fichier de profils trouve. Chemins testes : "
        + searched_paths
    )


def normalize_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les valeurs textuelles et prepare la table source."""
    _validate_required_columns(df)

    normalized = df.copy()
    normalized["id_profil"] = normalized["id_profil"].map(_normalize_profile_id)

    if normalized["id_profil"].eq("").any():
        raise ValueError("Certains identifiants de profil sont vides.")

    if normalized["id_profil"].duplicated().any():
        duplicated_ids = normalized.loc[
            normalized["id_profil"].duplicated(),
            "id_profil",
        ].tolist()
        raise ValueError(
            "Identifiants dupliques detectes : " + ", ".join(duplicated_ids)
        )

    return normalized


def encode_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les variables categorielle et ordinales du dataset."""
    normalized = normalize_profiles(df)
    encoded = normalized.copy()

    encoded["statut"] = _encode_series(
        normalized["statut"],
        STATUT_MAP,
        "statut",
    )
    encoded["specialite_dominante"] = _encode_series(
        normalized["specialite_dominante"],
        SPECIALITE_MAP,
        "specialite_dominante",
    )
    encoded["style_communication"] = _encode_series(
        normalized["style_communication"],
        STYLE_MAP,
        "style_communication",
    )

    for column_name in LIKERT_COLUMNS:
        encoded[column_name] = _encode_series(
            normalized[column_name],
            LIKERT_MAP,
            column_name,
        )

    return encoded


def prepare_processed_profiles(
    input_csv: str | None = None,
    output_csv: str = DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:
    """Construit et exporte le dataset traite a partir du brut."""
    source_csv = resolve_input_csv(input_csv)
    raw_df = load_profiles(source_csv)
    processed_df = encode_profiles(raw_df)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    return processed_df


def get_feature_columns() -> list[str]:
    """Retourne les colonnes utilisees pour le modele."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare le dataset de profils pour le pipeline ML.",
    )
    parser.add_argument(
        "--input",
        dest="input_csv",
        default=None,
        help="Fichier CSV source. Par defaut : raw_profiles.csv puis profiles_template.csv.",
    )
    parser.add_argument(
        "--output",
        dest="output_csv",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Fichier de sortie. Par defaut : {DEFAULT_OUTPUT_CSV}.",
    )
    args = parser.parse_args()

    source_csv = resolve_input_csv(args.input_csv)
    processed_df = prepare_processed_profiles(source_csv, args.output_csv)

    print("Dataset traite genere avec succes.")
    print(f"Source : {source_csv}")
    print(f"Sortie : {args.output_csv}")
    print(f"Profils prepares : {len(processed_df)}")


if __name__ == "__main__":
    main()
