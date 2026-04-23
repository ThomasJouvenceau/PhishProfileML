from __future__ import annotations

import os

from preprocess import encode_profiles, load_profiles
from recommend_category import apply_recommendations
from segment_profiles import (
    cluster_profiles,
    save_cluster_artifacts,
    summarize_clusters,
)


def main() -> None:
    input_csv = "data/profiles_template.csv"
    output_csv = "results/profiles_segmented.csv"

    os.makedirs("results", exist_ok=True)

    raw_df = load_profiles(input_csv)
    encoded_df = encode_profiles(raw_df)

    clustered_df, model, scaler = cluster_profiles(encoded_df, n_clusters=4)
    save_cluster_artifacts(model, scaler)

    final_df = apply_recommendations(clustered_df)
    final_df.to_csv(output_csv, index=False)

    print("=== Profils segmentés ===")
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

    print("\n=== Résumé des clusters ===")
    print(summarize_clusters(final_df))

    print(f"\nRésultats exportés dans : {output_csv}")


if __name__ == "__main__":
    main()
