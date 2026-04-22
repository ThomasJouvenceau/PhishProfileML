from __future__ import annotations

import pandas as pd


def recommend_message_category(row: pd.Series) -> str:
    """
    Recommande une catégorie de message à partir des caractéristiques du profil.
    Cette V1 utilise des règles simples et explicables.
    """

    if (
        row["implication_academique"] >= 3
        and row["interet_contenu_pedagogique"] >= 3
    ):
        return "pedagogique_institutionnel"

    if (
        row["implication_vie_etudiante"] >= 3
        and row["interet_evenementiel"] >= 3
    ):
        return "evenementiel_social"

    if (
        row["niveau_technique_estime"] >= 3
        and row["aisance_numerique"] >= 3
    ):
        return "technique_outils"

    if (
        row["style_communication"] == 2
        and row["presence_numerique_visible"] >= 2
    ):
        return "informel_communautaire"

    return "hybride_general"


def apply_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne de recommandation au dataset."""
    result = df.copy()
    result["categorie_message_recommandee"] = result.apply(
        recommend_message_category,
        axis=1,
    )
    return result