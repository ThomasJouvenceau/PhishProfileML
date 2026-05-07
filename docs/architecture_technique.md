# Architecture Technique

## Vue d'ensemble

Le projet suit une architecture simple en pipeline :

1. collecte et stockage des profils ;
2. preprocessing et encodage ;
3. segmentation ML ;
4. recommandation de categorie de message ;
5. evaluation experimentale ;
6. restitution via fichiers de resultats et app Streamlit.

## Composants principaux

### Donnees

- `data/raw_profiles.csv` : profils bruts
- `data/processed_profiles.csv` : profils encodes
- `data/campaign_results.csv` : resultats observes pendant l'experimentation

### Pipeline ML

- `src/preprocess.py` : validation, normalisation, encodage
- `src/segment_profiles.py` : clustering KMeans, score de confiance, resume des clusters
- `src/recommend_category.py` : recommandation de message
- `src/main.py` : orchestration du pipeline principal

### Evaluation

- `src/evaluate_results.py` : calcul des tableaux finaux et du resume JSON
- `results/tables/` : exports tabulaires
- `results/summary_metrics.json` : synthese structurée
- `results/final_summary.md` : conclusion redigee

### Visualisation

- `src/app.py` : application Streamlit de consultation
- `notebooks/exploration.ipynb` : exploration des profils
- `notebooks/evaluation.ipynb` : exploration des resultats

## Flux de donnees

```text
raw_profiles.csv
    -> preprocess.py
    -> processed_profiles.csv
    -> main.py
    -> profiles_segmented.csv
    -> evaluate_results.py + campaign_results.csv
    -> results/tables + summary_metrics.json + final_summary.md
    -> app.py
```

## Artefacts

- `models/kmeans_profiles.joblib`
- `models/scaler.joblib`

Ces fichiers sont regeneres par `src/main.py` et ignores par Git.
