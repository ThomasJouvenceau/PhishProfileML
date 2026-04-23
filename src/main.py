from __future__ import annotations

import os

from preprocess import prepare_processed_profiles, resolve_input_csv
from recommend_category import apply_recommendations
from segment_profiles import (
    cluster_profiles,
    save_cluster_artifacts,
    summarize_clusters,
)


def main() -> None:
    input_csv = resolve_input_csv()
    processed_csv = "data/processed_profiles.csv"
    output_csv = "results/profiles_segmented.csv"

    os.makedirs("results", exist_ok=True)

    processed_df = prepare_processed_profiles(input_csv, processed_csv)

    n_clusters = min(4, len(processed_df))
    if n_clusters < 2:
        raise ValueError(
            "Au moins 2 profils sont necessaires pour lancer la segmentation."
        )

    clustered_df, model, scaler = cluster_profiles(
        processed_df,
        n_clusters=n_clusters,
    )
    save_cluster_artifacts(model, scaler)

    final_df = apply_recommendations(clustered_df)
    final_df.to_csv(output_csv, index=False)

    print(f"Source profils : {input_csv}")
    print(f"Dataset traite exporte dans : {processed_csv}")
    print("=== Profils segmentes ===")
    print(
        final_df[
            [
                "id_profil",
                "cluster",
                "segment_principal",
                "score_confiance_segment",
                "categorie_message_recommandee",
            ]
        ]
    )

    print("\n=== Resume des clusters ===")
    print(summarize_clusters(final_df))

    print(f"\nResultats exportes dans : {output_csv}")


if __name__ == "__main__":
    main()
