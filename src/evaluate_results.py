from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd

from segment_profiles import summarize_clusters


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_SEGMENTED_CSV = BASE_DIR / "results" / "profiles_segmented.csv"
DEFAULT_CAMPAIGN_CSV = BASE_DIR / "data" / "campaign_results.csv"
DEFAULT_SUMMARY_JSON = BASE_DIR / "results" / "summary_metrics.json"
DEFAULT_FINAL_SUMMARY_MD = BASE_DIR / "results" / "final_summary.md"
DEFAULT_TABLES_DIR = BASE_DIR / "results" / "tables"

METRIC_COLUMNS = [
    "taux_ouverture",
    "taux_clic",
    "taux_reponse",
    "delai_reaction_moyen_heures",
]

CONDITION_ALIASES = {
    "condition": "condition",
    "type_campagne": "condition",
    "campagne": "condition",
    "approche": "condition",
}

RATE_ALIASES = {
    "ouvert": "taux_ouverture",
    "clique": "taux_clic",
    "repondu": "taux_reponse",
}

CONDITION_VALUE_ALIASES = {
    "generique": "generique",
    "generic": "generique",
    "campagne_generique": "generique",
    "personnalisee": "personnalisee",
    "personalisee": "personnalisee",
    "personnalized": "personnalisee",
    "personalized": "personnalisee",
    "campagne_personnalisee": "personnalisee",
}


def _normalize_token(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    return pd.read_csv(path)


def _prepare_output_dirs(tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.parent.mkdir(parents=True, exist_ok=True)


def _rename_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()

    column_mapping: dict[str, str] = {}
    for column in renamed.columns:
        normalized = _normalize_token(column)
        if normalized in CONDITION_ALIASES:
            column_mapping[column] = CONDITION_ALIASES[normalized]
        elif normalized in RATE_ALIASES:
            column_mapping[column] = RATE_ALIASES[normalized]

    if column_mapping:
        renamed = renamed.rename(columns=column_mapping)

    return renamed


def _normalize_condition_value(value: object) -> str:
    normalized = _normalize_token(value).replace(" ", "_")
    return CONDITION_VALUE_ALIASES.get(normalized, normalized)


def _extract_long_campaign_results(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, list[str]]:
    """
    Retourne les resultats de campagne au format long standard.

    Format standard cible :
    - id_profil
    - condition
    - taux_ouverture
    - taux_clic
    - taux_reponse
    - delai_reaction_moyen_heures
    """

    normalized = _rename_known_columns(df)
    warnings: list[str] = []

    if "id_profil" not in normalized.columns:
        raise ValueError("La colonne 'id_profil' est obligatoire dans campaign_results.csv.")

    if "condition" in normalized.columns:
        result = normalized.copy()
        result["condition"] = result["condition"].map(_normalize_condition_value)

        existing_metrics = [column for column in METRIC_COLUMNS if column in result.columns]
        if not existing_metrics:
            raise ValueError(
                "Aucune metrique compatible n'a ete trouvee dans campaign_results.csv."
            )

        columns_to_keep = ["id_profil", "condition", *existing_metrics]
        result = result[columns_to_keep]

        for metric_column in existing_metrics:
            result[metric_column] = pd.to_numeric(result[metric_column], errors="coerce")

        return result, True, warnings

    wide_frames: list[pd.DataFrame] = []
    for condition in ("generique", "personnalisee"):
        mapped_metrics: dict[str, str] = {}
        for metric_name in METRIC_COLUMNS:
            candidate_columns = (
                f"{condition}_{metric_name}",
                f"{metric_name}_{condition}",
            )
            for candidate in candidate_columns:
                if candidate in normalized.columns:
                    mapped_metrics[metric_name] = candidate
                    break

        if mapped_metrics:
            frame = normalized[["id_profil", *mapped_metrics.values()]].copy()
            frame = frame.rename(columns={source: target for target, source in mapped_metrics.items()})
            frame["condition"] = condition
            wide_frames.append(frame)

    if wide_frames:
        result = pd.concat(wide_frames, ignore_index=True, sort=False)
        for metric_column in METRIC_COLUMNS:
            if metric_column in result.columns:
                result[metric_column] = pd.to_numeric(result[metric_column], errors="coerce")
        return result, True, warnings

    existing_metrics = [column for column in METRIC_COLUMNS if column in normalized.columns]
    if existing_metrics:
        warnings.append(
            "campaign_results.csv ne contient pas de colonne de condition "
            "(generique/personnalisee). Les comparaisons par campagne seront marquees comme indisponibles."
        )

        result = normalized[["id_profil", *existing_metrics]].copy()
        result["condition"] = "globale"

        for metric_column in existing_metrics:
            result[metric_column] = pd.to_numeric(result[metric_column], errors="coerce")

        return result, False, warnings

    raise ValueError(
        "Format de campaign_results.csv non reconnu. "
        "Attendu : soit une colonne 'condition', soit des colonnes suffixees/prefixees "
        "par condition, soit des metriques globales."
    )


def extract_long_campaign_results(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, list[str]]:
    """Expose le normaliseur de resultats de campagne pour les notebooks."""
    return _extract_long_campaign_results(df)


def _round_value(value: object, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _mode_or_empty(series: pd.Series) -> str:
    if series.empty:
        return ""

    modes = series.mode(dropna=True)
    if modes.empty:
        return ""
    return str(modes.iloc[0])


def build_cluster_summary(segmented_df: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_clusters(segmented_df).reset_index()

    if "categorie_message_recommandee" in segmented_df.columns:
        dominant_categories = (
            segmented_df.groupby("cluster")["categorie_message_recommandee"]
            .agg(_mode_or_empty)
            .rename("categorie_message_dominante")
            .reset_index()
        )
        summary = summary.merge(dominant_categories, on="cluster", how="left")

    if "segment_principal" in summary.columns:
        summary["interpretation"] = summary["segment_principal"].map(
            {
                "profil_academique": "Interet marque pour les contenus pedagogiques.",
                "profil_associatif_evenementiel": "Sensibilite forte a la vie etudiante et aux evenements.",
                "profil_technique_vigilant": "Aisance numerique et orientation technique elevees.",
                "profil_hybride_social": "Profil equilibre avec visibilite sociale plus marquee.",
            }
        ).fillna("")
    else:
        summary["interpretation"] = ""

    return summary


def build_recommendation_distribution(segmented_df: pd.DataFrame) -> pd.DataFrame:
    if "categorie_message_recommandee" not in segmented_df.columns:
        return pd.DataFrame(
            columns=[
                "categorie_message_recommandee",
                "nb_profils",
                "part_du_total",
                "segment_majoritaire",
                "commentaire",
            ]
        )

    total_profiles = max(len(segmented_df), 1)
    grouped = (
        segmented_df.groupby("categorie_message_recommandee")
        .agg(
            nb_profils=("id_profil", "count"),
            segment_majoritaire=("segment_principal", _mode_or_empty),
        )
        .reset_index()
    )
    grouped["part_du_total"] = (grouped["nb_profils"] / total_profiles).round(4)
    grouped["commentaire"] = ""
    return grouped


def _aggregate_metrics(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    available_metrics = [column for column in METRIC_COLUMNS if column in df.columns]

    aggregated = (
        df.groupby(group_columns)
        .agg(
            nb_profils=("id_profil", "nunique"),
            **{metric_name: (metric_name, "mean") for metric_name in available_metrics},
        )
        .reset_index()
    )

    if {"taux_ouverture", "taux_clic", "taux_reponse"}.issubset(aggregated.columns):
        aggregated["performance_globale"] = (
            aggregated[["taux_ouverture", "taux_clic", "taux_reponse"]]
            .mean(axis=1)
            .round(4)
        )
    else:
        aggregated["performance_globale"] = pd.NA

    for column in available_metrics:
        aggregated[column] = aggregated[column].round(4)

    return aggregated


def build_campaign_outputs(
    segmented_df: pd.DataFrame,
    campaign_results_df: pd.DataFrame,
    can_compare_conditions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    merged = campaign_results_df.merge(
        segmented_df,
        on="id_profil",
        how="left",
        indicator=True,
    )

    unmatched_profiles = (
        merged.loc[merged["_merge"] != "both", "id_profil"]
        .dropna()
        .unique()
        .tolist()
    )
    merged = merged.drop(columns="_merge")

    if unmatched_profiles:
        merged = merged[merged["segment_principal"].notna()].copy()

    available_conditions = sorted(campaign_results_df["condition"].dropna().unique().tolist())

    if can_compare_conditions and {"generique", "personnalisee"}.issubset(available_conditions):
        campaign_comparison = _aggregate_metrics(campaign_results_df, ["condition"])
        campaign_comparison["commentaire"] = ""

        if not merged.empty:
            performance_by_segment = _aggregate_metrics(
                merged,
                ["segment_principal", "condition"],
            )
            performance_by_segment["commentaire"] = ""
        else:
            performance_by_segment = pd.DataFrame(
                columns=[
                    "segment_principal",
                    "condition",
                    "nb_profils",
                    "taux_ouverture",
                    "taux_clic",
                    "taux_reponse",
                    "delai_reaction_moyen_heures",
                    "performance_globale",
                    "commentaire",
                ]
            )

        generique_row = campaign_comparison.loc[
            campaign_comparison["condition"] == "generique"
        ].iloc[0]
        personnalisee_row = campaign_comparison.loc[
            campaign_comparison["condition"] == "personnalisee"
        ].iloc[0]

        score_generique = generique_row.get("taux_clic")
        score_personnalise = personnalisee_row.get("taux_clic")
        delta_absolu = (
            None
            if pd.isna(score_generique) or pd.isna(score_personnalise)
            else round(float(score_personnalise - score_generique), 4)
        )
        delta_relatif = (
            None
            if delta_absolu is None or score_generique in (0, None) or pd.isna(score_generique)
            else round(float(delta_absolu / score_generique) * 100, 2)
        )
        hypothesis_validated = delta_absolu is not None and delta_absolu > 0

        hypothesis_check = pd.DataFrame(
            [
                {
                    "metrique_principale": "taux_clic",
                    "score_generique": _round_value(score_generique),
                    "score_personnalise": _round_value(score_personnalise),
                    "delta_absolu": delta_absolu,
                    "delta_relatif_pourcent": delta_relatif,
                    "hypothese_validee": hypothesis_validated,
                    "commentaire": "",
                }
            ]
        )

        if not performance_by_segment.empty and "taux_clic" in performance_by_segment.columns:
            pivot = (
                performance_by_segment.pivot(
                    index="segment_principal",
                    columns="condition",
                    values="taux_clic",
                )
                .reset_index()
                .rename_axis(None, axis=1)
            )
            if {"generique", "personnalisee"}.issubset(pivot.columns):
                pivot["delta_taux_clic"] = (
                    pivot["personnalisee"] - pivot["generique"]
                ).round(4)
                most_responsive_segment = (
                    pivot.sort_values("delta_taux_clic", ascending=False)
                    .iloc[0]["segment_principal"]
                )
                performance_by_segment_summary = pivot.to_dict(orient="records")
            else:
                most_responsive_segment = ""
                performance_by_segment_summary = []
        else:
            most_responsive_segment = ""
            performance_by_segment_summary = []

        summary_data = {
            "comparison_available": True,
            "unmatched_profiles": unmatched_profiles,
            "campaign_comparison": {
                "generique": {
                    key: _round_value(generique_row.get(key))
                    for key in [
                        "taux_ouverture",
                        "taux_clic",
                        "taux_reponse",
                        "delai_reaction_moyen_heures",
                        "performance_globale",
                    ]
                },
                "personnalisee": {
                    key: _round_value(personnalisee_row.get(key))
                    for key in [
                        "taux_ouverture",
                        "taux_clic",
                        "taux_reponse",
                        "delai_reaction_moyen_heures",
                        "performance_globale",
                    ]
                },
                "delta_personnalisation": {
                    key: (
                        None
                        if pd.isna(generique_row.get(key)) or pd.isna(personnalisee_row.get(key))
                        else round(float(personnalisee_row.get(key) - generique_row.get(key)), 4)
                    )
                    for key in [
                        "taux_ouverture",
                        "taux_clic",
                        "taux_reponse",
                        "delai_reaction_moyen_heures",
                        "performance_globale",
                    ]
                },
            },
            "performance_by_segment": performance_by_segment_summary,
            "hypothesis_validated": hypothesis_validated,
            "most_responsive_segment": most_responsive_segment,
        }

        return (
            campaign_comparison,
            performance_by_segment,
            hypothesis_check,
            summary_data,
        )

    campaign_comparison = _aggregate_metrics(campaign_results_df, ["condition"])
    campaign_comparison["commentaire"] = (
        "Comparaison generique/personnalisee indisponible avec le format actuel."
    )

    if not merged.empty:
        performance_by_segment = _aggregate_metrics(
            merged,
            ["segment_principal", "condition"],
        )
        performance_by_segment["commentaire"] = (
            "Analyse calculee sans separation explicite des conditions."
        )
    else:
        performance_by_segment = pd.DataFrame(
            columns=[
                "segment_principal",
                "condition",
                "nb_profils",
                "taux_ouverture",
                "taux_clic",
                "taux_reponse",
                "delai_reaction_moyen_heures",
                "performance_globale",
                "commentaire",
            ]
        )

    hypothesis_check = pd.DataFrame(
        [
            {
                "metrique_principale": "taux_clic",
                "score_generique": pd.NA,
                "score_personnalise": pd.NA,
                "delta_absolu": pd.NA,
                "delta_relatif_pourcent": pd.NA,
                "hypothese_validee": pd.NA,
                "commentaire": (
                    "Impossible de tester l'hypothese : "
                    "la colonne de condition generique/personnalisee est absente."
                ),
            }
        ]
    )

    summary_data = {
        "comparison_available": False,
        "unmatched_profiles": unmatched_profiles,
        "campaign_comparison": {
            "generique": {key: None for key in METRIC_COLUMNS + ["performance_globale"]},
            "personnalisee": {key: None for key in METRIC_COLUMNS + ["performance_globale"]},
            "delta_personnalisation": {
                key: None for key in METRIC_COLUMNS + ["performance_globale"]
            },
        },
        "performance_by_segment": [],
        "hypothesis_validated": None,
        "most_responsive_segment": "",
    }

    return campaign_comparison, performance_by_segment, hypothesis_check, summary_data


def build_summary_json(
    segmented_df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    campaign_summary: dict,
    warnings: list[str],
) -> dict:
    cluster_items = []
    if not cluster_summary.empty:
        for row in cluster_summary.to_dict(orient="records"):
            cluster_items.append(
                {
                    "cluster": int(row["cluster"]),
                    "segment_principal": row.get("segment_principal", ""),
                    "nb_profils": (
                        None if pd.isna(row.get("nb_profils")) else int(row["nb_profils"])
                    ),
                    "score_confiance_moyen": _round_value(
                        row.get("score_confiance_moyen"),
                        digits=3,
                    ),
                }
            )

    performance_by_segment = campaign_summary.get("performance_by_segment", [])
    summary_json = {
        "project_name": "PhishProfileML",
        "primary_metric": "taux_clic",
        "dataset": {
            "nb_profils_total": int(len(segmented_df)),
            "nb_profils_segmentes": int(len(segmented_df)),
            "nb_segments": int(segmented_df["cluster"].nunique()) if "cluster" in segmented_df.columns else None,
            "source_profiles": "data/processed_profiles.csv",
            "source_campaign_results": "data/campaign_results.csv",
            "date_export": pd.Timestamp.now().date().isoformat(),
            "comparison_available": campaign_summary["comparison_available"],
        },
        "segmentation": {
            "score_confiance_moyen_global": (
                _round_value(segmented_df["score_confiance_segment"].mean(), digits=3)
                if "score_confiance_segment" in segmented_df.columns
                else None
            ),
            "clusters": cluster_items,
        },
        "campaign_comparison": campaign_summary["campaign_comparison"],
        "performance_by_segment": [
            {
                "segment_principal": row.get("segment_principal", ""),
                "taux_clic_generique": _round_value(row.get("generique")),
                "taux_clic_personnalise": _round_value(row.get("personnalisee")),
                "delta_taux_clic": _round_value(row.get("delta_taux_clic")),
            }
            for row in performance_by_segment
        ],
        "conclusion": {
            "hypothesis_validated": campaign_summary["hypothesis_validated"],
            "main_finding": (
                "La personnalisation ameliore le taux de clic."
                if campaign_summary["hypothesis_validated"] is True
                else (
                    "Impossible de conclure sur l'effet de la personnalisation."
                    if campaign_summary["hypothesis_validated"] is None
                    else "La personnalisation n'ameliore pas le taux de clic dans les donnees actuelles."
                )
            ),
            "most_responsive_segment": campaign_summary["most_responsive_segment"],
            "main_limitations": warnings
            + (
                [
                    "Certains profils de campagne ne correspondent pas au fichier de segmentation courant."
                ]
                if campaign_summary.get("unmatched_profiles")
                else []
            ),
        },
    }

    return summary_json


def build_final_summary_markdown(summary_json: dict) -> str:
    conclusion = summary_json["conclusion"]
    comparison = summary_json["campaign_comparison"]

    generique_click = comparison["generique"].get("taux_clic")
    personnalisee_click = comparison["personnalisee"].get("taux_clic")
    delta_click = comparison["delta_personnalisation"].get("taux_clic")

    lines = [
        "# Synthese Finale",
        "",
        "## Hypothese",
        "",
        "Une campagne personnalisee basee sur la segmentation ML performe mieux qu'une campagne generique.",
        "",
        "## Resultat principal",
        "",
        conclusion["main_finding"],
        "",
        "## Comparaison generique vs personnalisee",
        "",
        f"- Taux de clic generique : {generique_click}",
        f"- Taux de clic personnalise : {personnalisee_click}",
        f"- Delta taux de clic : {delta_click}",
        "",
        "## Segments les plus reactifs",
        "",
        conclusion["most_responsive_segment"] or "Non determine avec les donnees actuelles.",
        "",
        "## Coherence des recommandations",
        "",
        "A confronter avec les performances observees par segment et par categorie de message.",
        "",
        "## Limites de l'experience",
        "",
    ]

    limitations = conclusion.get("main_limitations", [])
    if limitations:
        lines.extend([f"- {limitation}" for limitation in limitations])
    else:
        lines.append("- Aucune limite supplementaire n'a ete enregistree automatiquement.")

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Ce resume doit etre relu et adapte par l'equipe avant le rendu final.",
            "",
        ]
    )

    return "\n".join(lines)


def run_evaluation(
    segmented_csv: Path = DEFAULT_SEGMENTED_CSV,
    campaign_csv: Path = DEFAULT_CAMPAIGN_CSV,
    summary_json_path: Path = DEFAULT_SUMMARY_JSON,
    final_summary_path: Path = DEFAULT_FINAL_SUMMARY_MD,
    tables_dir: Path = DEFAULT_TABLES_DIR,
) -> None:
    _prepare_output_dirs(tables_dir)

    segmented_df = _load_csv(segmented_csv)
    campaign_raw_df = _load_csv(campaign_csv)

    campaign_results_df, can_compare_conditions, warnings = _extract_long_campaign_results(
        campaign_raw_df
    )

    cluster_summary = build_cluster_summary(segmented_df)
    recommendation_distribution = build_recommendation_distribution(segmented_df)
    campaign_comparison, performance_by_segment, hypothesis_check, campaign_summary = (
        build_campaign_outputs(
            segmented_df,
            campaign_results_df,
            can_compare_conditions=can_compare_conditions,
        )
    )

    cluster_summary.to_csv(tables_dir / "cluster_summary.csv", index=False)
    recommendation_distribution.to_csv(
        tables_dir / "recommendation_distribution.csv",
        index=False,
    )
    campaign_comparison.to_csv(tables_dir / "campaign_comparison.csv", index=False)
    performance_by_segment.to_csv(
        tables_dir / "performance_by_segment.csv",
        index=False,
    )
    hypothesis_check.to_csv(tables_dir / "hypothesis_check.csv", index=False)

    summary_json = build_summary_json(
        segmented_df=segmented_df,
        cluster_summary=cluster_summary,
        campaign_summary=campaign_summary,
        warnings=warnings,
    )
    summary_json_path.write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    final_summary_path.write_text(
        build_final_summary_markdown(summary_json),
        encoding="utf-8",
    )

    print(f"Evaluation terminee. Resultats ecrits dans : {tables_dir.parent}")
    if warnings:
        print("Avertissements :")
        for warning in warnings:
            print(f"- {warning}")
    if campaign_summary.get("unmatched_profiles"):
        print("Profils de campagne absents de la segmentation courante :")
        for profile_id in campaign_summary["unmatched_profiles"]:
            print(f"- {profile_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evalue les resultats de campagne et genere les livrables dans results/.",
    )
    parser.add_argument(
        "--segmented",
        default=str(DEFAULT_SEGMENTED_CSV),
        help=f"CSV des profils segmentes. Par defaut : {DEFAULT_SEGMENTED_CSV}",
    )
    parser.add_argument(
        "--campaign",
        default=str(DEFAULT_CAMPAIGN_CSV),
        help=f"CSV des resultats de campagne. Par defaut : {DEFAULT_CAMPAIGN_CSV}",
    )
    args = parser.parse_args()

    run_evaluation(
        segmented_csv=Path(args.segmented),
        campaign_csv=Path(args.campaign),
    )


if __name__ == "__main__":
    main()
