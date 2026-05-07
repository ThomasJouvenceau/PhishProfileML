# Protocole Experimental

## Objectif

Verifier si une campagne email personnalisee a partir d'une segmentation ML obtient de meilleurs resultats qu'une campagne generique.

## Hypothese

La personnalisation ameliore en priorite le taux de clic, puis potentiellement le taux d'ouverture et le taux de reponse.

## Population

- 26 etudiants de derniere annee de bachelor
- meme promotion
- identifiants anonymises de type `P01`, `P02`, etc.

## Design experimental

Le protocole cible est un protocole croise :

1. chaque participant recoit une campagne generique ;
2. a un autre moment, le meme participant recoit une campagne personnalisee ;
3. les deux conditions sont comparees sur le meme panel.

## Donnees necessaires

### Profils

Fichier source : `data/raw_profiles.csv`

Variables attendues :
- `id_profil`
- `statut`
- `specialite_dominante`
- `implication_academique`
- `interet_contenu_pedagogique`
- `implication_vie_etudiante`
- `interet_evenementiel`
- `niveau_technique_estime`
- `aisance_numerique`
- `style_communication`
- `reactivite_percue`
- `presence_numerique_visible`

### Resultats de campagne

Fichier source : `data/campaign_results.csv`

Format recommande pour la comparaison :
- `id_profil`
- `condition`
- `taux_ouverture`
- `taux_clic`
- `taux_reponse`
- `delai_reaction_moyen_heures`

Valeurs attendues dans `condition` :
- `generique`
- `personnalisee`

## Metriques

Metrique principale :
- `taux_clic`

Metriques secondaires :
- `taux_ouverture`
- `taux_reponse`
- `delai_reaction_moyen_heures`
- `performance_globale`

## Etapes d'analyse

1. preparer le dataset traite avec `src/preprocess.py` ;
2. segmenter les profils avec `src/main.py` ;
3. evaluer les resultats avec `src/evaluate_results.py` ;
4. verifier les tableaux dans `results/tables/` ;
5. synthetiser la conclusion dans `results/final_summary.md`.

## Limites

- panel reduit ;
- variables de profil partiellement subjectives ;
- contexte academique non generalisable ;
- resultats sensibles a la qualite de `campaign_results.csv`.
