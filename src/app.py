from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PROFILES_PATH = BASE_DIR / "data" / "processed_profiles.csv"
CAMPAIGN_RESULTS_PATH = BASE_DIR / "data" / "campaign_results.csv"
SEGMENTED_PROFILES_PATH = BASE_DIR / "results" / "profiles_segmented.csv"
SUMMARY_METRICS_PATH = BASE_DIR / "results" / "summary_metrics.json"
FINAL_SUMMARY_PATH = BASE_DIR / "results" / "final_summary.md"

TABLES_DIR = BASE_DIR / "results" / "tables"
CHARTS_DIR = BASE_DIR / "results" / "charts"


st.set_page_config(
    page_title="PhishProfileML",
    page_icon="🎯",
    layout="wide",
)


def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    return None


def load_markdown(path: Path) -> str | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return file.read()
    return None


def format_metric_value(value: object, *, percentage: bool = False, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"

    numeric_value = float(value)
    if percentage:
        return f"{numeric_value * 100:.1f}%"
    if suffix:
        return f"{numeric_value:.2f} {suffix}"
    return f"{numeric_value:.3f}"


def render_metric_group(title: str, metrics: dict[str, object]) -> None:
    st.subheader(title)
    columns = st.columns(5)
    metric_specs = [
        ("taux_ouverture", "Taux ouverture", True, ""),
        ("taux_clic", "Taux clic", True, ""),
        ("taux_reponse", "Taux reponse", True, ""),
        ("delai_reaction_moyen_heures", "Delai moyen", False, "h"),
        ("performance_globale", "Performance globale", True, ""),
    ]

    for index, (key, label, percentage, suffix) in enumerate(metric_specs):
        with columns[index]:
            st.metric(
                label=label,
                value=format_metric_value(
                    metrics.get(key),
                    percentage=percentage,
                    suffix=suffix,
                ),
            )


st.title("PhishProfileML")
st.subheader("Segmentation ML de profils étudiants et recommandation de catégories de messages")

st.markdown(
    """
PhishProfileML est un projet de machine learning appliqué à la cybersécurité.

L’objectif est de segmenter des profils étudiants anonymisés afin de recommander
des catégories de messages personnalisés et de comparer une campagne générique
avec une campagne personnalisée.
"""
)

st.divider()

segmented_df = load_csv(SEGMENTED_PROFILES_PATH)
processed_df = load_csv(PROCESSED_PROFILES_PATH)
campaign_df = load_csv(CAMPAIGN_RESULTS_PATH)
summary_metrics = load_json(SUMMARY_METRICS_PATH)
final_summary = load_markdown(FINAL_SUMMARY_PATH)

if segmented_df is None:
    st.error(
        "Le fichier results/profiles_segmented.csv est introuvable. "
        "Lance d'abord le pipeline avec : python src/main.py"
    )
    st.stop()

if campaign_df is not None and len(segmented_df) != len(campaign_df):
    st.warning(
        "Le nombre de profils segmentes ne correspond pas au nombre de lignes "
        "dans data/campaign_results.csv. Les sorties doivent probablement etre regenerees."
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Profils segmentés",
        "Segments",
        "Recommandations",
        "Résultats",
        "Résumé final",
    ]
)

with tab1:
    st.header("Profils segmentés")

    st.write("Aperçu du fichier généré par le pipeline ML :")
    st.dataframe(segmented_df, use_container_width=True)

    st.write("Dimensions du dataset :")
    col1, col2 = st.columns(2)
    col1.metric("Nombre de profils", segmented_df.shape[0])
    col2.metric("Nombre de variables", segmented_df.shape[1])

    if processed_df is not None:
        with st.expander("Voir le dataset traité"):
            st.dataframe(processed_df, use_container_width=True)

with tab2:
    st.header("Analyse des segments")

    if "cluster" in segmented_df.columns:
        cluster_counts = segmented_df["cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)

        st.write("Répartition des profils par cluster :")
        st.dataframe(
            cluster_counts.rename("nombre_profils").reset_index().rename(
                columns={"index": "cluster"}
            ),
            use_container_width=True,
        )
    else:
        st.warning("La colonne 'cluster' n'existe pas dans profiles_segmented.csv.")

    if "segment_principal" in segmented_df.columns:
        st.write("Répartition par segment principal :")
        segment_counts = segmented_df["segment_principal"].value_counts()
        st.bar_chart(segment_counts)

        selected_segment = st.selectbox(
            "Filtrer par segment principal :",
            ["Tous"] + sorted(segmented_df["segment_principal"].dropna().unique().tolist()),
        )

        if selected_segment != "Tous":
            filtered_df = segmented_df[
                segmented_df["segment_principal"] == selected_segment
            ]
            st.dataframe(filtered_df, use_container_width=True)
        else:
            st.dataframe(segmented_df, use_container_width=True)
    else:
        st.warning("La colonne 'segment_principal' n'existe pas dans profiles_segmented.csv.")

with tab3:
    st.header("Recommandation de catégories de messages")

    required_columns = [
        "id_profil",
        "cluster",
        "segment_principal",
        "score_confiance_segment",
        "categorie_message_recommandee",
    ]

    existing_columns = [
        column for column in required_columns if column in segmented_df.columns
    ]

    if existing_columns:
        st.dataframe(segmented_df[existing_columns], use_container_width=True)
    else:
        st.warning("Les colonnes de recommandation attendues ne sont pas disponibles.")

    if "categorie_message_recommandee" in segmented_df.columns:
        st.write("Répartition des catégories recommandées :")
        category_counts = segmented_df["categorie_message_recommandee"].value_counts()
        st.bar_chart(category_counts)

with tab4:
    st.header("Résultats de campagne")

    if summary_metrics is not None:
        dataset_info = summary_metrics.get("dataset", {})
        segmentation_info = summary_metrics.get("segmentation", {})
        campaign_comparison = summary_metrics.get("campaign_comparison", {})
        conclusion = summary_metrics.get("conclusion", {})

        st.subheader("Synthese")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Profils segmentes",
            dataset_info.get("nb_profils_segmentes", "N/A"),
        )
        col2.metric(
            "Segments detectes",
            dataset_info.get("nb_segments", "N/A"),
        )
        col3.metric(
            "Confiance moyenne",
            format_metric_value(
                segmentation_info.get("score_confiance_moyen_global"),
            ),
        )

        generique_metrics = campaign_comparison.get("generique", {})
        personnalisee_metrics = campaign_comparison.get("personnalisee", {})
        delta_metrics = campaign_comparison.get("delta_personnalisation", {})
        comparison_available = dataset_info.get("comparison_available")
        if comparison_available is None:
            comparison_available = (
                generique_metrics.get("taux_clic") is not None
                and personnalisee_metrics.get("taux_clic") is not None
            )

        if comparison_available:
            render_metric_group("Campagne generique", generique_metrics)
            render_metric_group("Campagne personnalisee", personnalisee_metrics)
            render_metric_group("Delta personnalisation", delta_metrics)
        else:
            st.info(
                "La comparaison generique vs personnalisee n'est pas encore disponible "
                "avec le format actuel de campaign_results.csv."
            )

        limitations = conclusion.get("main_limitations", [])
        if limitations:
            st.subheader("Points de vigilance")
            for limitation in limitations:
                st.write(f"- {limitation}")

        main_finding = conclusion.get("main_finding")
        if main_finding:
            st.subheader("Lecture rapide")
            st.write(main_finding)

        with st.expander("Voir le fichier summary_metrics.json"):
            st.json(summary_metrics)
    else:
        st.info(
            "Aucun fichier summary_metrics.json trouvé. "
            "Les résultats expérimentaux seront affichés ici lorsqu'ils seront disponibles."
        )

    if TABLES_DIR.exists():
        table_files = sorted(TABLES_DIR.glob("*.csv"))

        if table_files:
            st.subheader("Tables de résultats")

            selected_table = st.selectbox(
                "Choisir une table :",
                [file.name for file in table_files],
            )

            table_path = TABLES_DIR / selected_table
            table_df = pd.read_csv(table_path)
            st.dataframe(table_df, use_container_width=True)

    if CHARTS_DIR.exists():
        chart_files = sorted(
            list(CHARTS_DIR.glob("*.png"))
            + list(CHARTS_DIR.glob("*.jpg"))
            + list(CHARTS_DIR.glob("*.jpeg"))
        )

        if chart_files:
            st.subheader("Graphiques")

            for chart_file in chart_files:
                st.image(str(chart_file), caption=chart_file.name)

with tab5:
    st.header("Résumé final")

    if final_summary is not None:
        st.markdown(final_summary)
    else:
        st.info(
            "Aucun fichier final_summary.md trouvé. "
            "Le résumé final sera affiché ici lorsqu'il sera disponible."
        )
