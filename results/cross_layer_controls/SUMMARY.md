# Contrôles croisés de couches : Tri et Qwen

Date : 2026-09-14  
Modèles : `trillionlabs/Tri-0.5B-Base` et `Qwen/Qwen2.5-0.5B`  
Protocole : mêmes splits FLORES, mêmes 4 langues, mêmes poolings, même sous-espace de dimension 3 et mêmes cinq bases aléatoires que dans l'expérience principale.

## Question

L'analyse principale étudiait le bloc 7 de Tri et le bloc 22 de Qwen. Comme les deux modèles ont 24 blocs, le contrôle strict consiste à ajouter Qwen au bloc 7 et Tri au bloc 22.

## Intervention : retrieval final à la couche 24

Les valeurs sont les R@1 moyens sur les 12 directions linguistiques. Le gain et l'intervalle de confiance bootstrap à 95 % comparent le retrait du sous-espace linguistique au forward normal.

| Modèle | Bloc d'intervention | Pooling | Normal | Retrait langue | Gain | IC 95 % du gain |
|---|---:|---|---:|---:|---:|---:|
| Tri | 7 | mean | 0,1242 | 0,3252 | +0,2010 | [0,1846 ; 0,2165] |
| Tri | 22 | mean | 0,1242 | 0,3662 | +0,2420 | [0,2262 ; 0,2575] |
| Tri | 7 | last token | 0,2324 | 0,3102 | +0,0778 | [0,0609 ; 0,0954] |
| Tri | 22 | last token | 0,2324 | 0,2271 | -0,0054 | [-0,0203 ; 0,0093] |
| Qwen | 7 | mean | 0,0433 | 0,1950 | +0,1517 | [0,1421 ; 0,1613] |
| Qwen | 22 | mean | 0,0433 | 0,7062 | +0,6629 | [0,6436 ; 0,6834] |
| Qwen | 7 | last token | 0,0257 | 0,1326 | +0,1069 | [0,0973 ; 0,1175] |
| Qwen | 22 | last token | 0,0257 | 0,3662 | +0,3405 | [0,3255 ; 0,3564] |

## Décomposition intra-bloc

Les chiffres suivants comparent la sortie de l'attention à la sortie du bloc, donc isolent principalement l'effet du MLP.

| Modèle | Bloc | Pooling | R@1 brut post-attention → sortie | Fraction de variance linguistique post-attention → sortie |
|---|---:|---|---:|---:|
| Tri | 7 | mean | 0,8169 → 0,2909 | 0,2294 → 0,9831 |
| Tri | 22 | mean | 0,4056 → 0,3171 | 0,7524 → 0,6975 |
| Qwen | 7 | mean | 0,4762 → 0,4624 | 0,9607 → 0,9605 |
| Qwen | 22 | mean | 0,4064 → 0,5184 | 0,5773 → 0,4713 |
| Tri | 7 | last token | 0,4995 → 0,4334 | 0,2380 → 0,2383 |
| Tri | 22 | last token | 0,6789 → 0,6458 | 0,3566 → 0,4192 |
| Qwen | 7 | last token | 0,3411 → 0,2726 | 0,3342 → 0,3276 |
| Qwen | 22 | last token | 0,6346 → 0,6050 | 0,4622 → 0,4720 |

## Contrôle PCA

À la couche finale, le retrait de la base PCA donne :

| Modèle | Bloc | Pooling | Retrait langue | Retrait PCA | Langue - PCA, IC 95 % |
|---|---:|---|---:|---:|---:|
| Tri | 7 | mean | 0,3252 | 0,1478 | +0,1774 [0,1605 ; 0,1927] |
| Tri | 22 | mean | 0,3662 | 0,3276 | +0,0386 [0,0247 ; 0,0542] |
| Tri | 7 | last token | 0,3102 | 0,2147 | +0,0955 [0,0788 ; 0,1120] |
| Tri | 22 | last token | 0,2271 | 0,1701 | +0,0570 [0,0430 ; 0,0703] |
| Qwen | 7 | mean | 0,1950 | 0,3328 | -0,1379 [-0,1548 ; -0,1214] |
| Qwen | 22 | mean | 0,7062 | 0,6862 | +0,0200 [0,0145 ; 0,0260] |
| Qwen | 7 | last token | 0,1326 | 0,1361 | -0,0034 [-0,0169 ; 0,0107] |
| Qwen | 22 | last token | 0,3662 | 0,3566 | +0,0096 [0,0033 ; 0,0155] |

## Extension MASSIVE

Le tableau rapporte l'accuracy moyenne de transfert depuis l'anglais vers KO, JA et ZH. Les intervalles bootstrap à 95 % portent sur le gain de la base linguistique par rapport au forward normal.

| Modèle | Bloc | Pooling | Normal | Retrait langue | Gain | IC 95 % du gain |
|---|---:|---|---:|---:|---:|---:|
| Tri | 7 | mean | 0,2971 | 0,4436 | +0,1464 | [0,1304 ; 0,1622] |
| Tri | 22 | mean | 0,2971 | 0,5071 | +0,2100 | [0,1940 ; 0,2264] |
| Tri | 7 | last token | 0,0713 | 0,3671 | +0,2958 | [0,2811 ; 0,3122] |
| Tri | 22 | last token | 0,0713 | 0,3389 | +0,2676 | [0,2520 ; 0,2840] |
| Qwen | 7 | mean | 0,2636 | 0,4769 | +0,2133 | [0,1982 ; 0,2291] |
| Qwen | 22 | mean | 0,2636 | 0,4200 | +0,1564 | [0,1444 ; 0,1687] |
| Qwen | 7 | last token | 0,3240 | 0,3782 | +0,0542 | [0,0378 ; 0,0709] |
| Qwen | 22 | last token | 0,3240 | 0,3651 | +0,0411 | [0,0296 ; 0,0531] |

Sur MASSIVE, l'ordre des couches n'est donc pas le même que sur le retrieval FLORES : Tri préfère le bloc 22 en mean pooling mais légèrement le bloc 7 en last-token pooling ; Qwen préfère le bloc 7 avec les deux poolings. Tous les gains de la base linguistique contre le forward normal sont positifs et leurs intervalles excluent zéro.

## Extension LM : coût en NLL

La NLL normale est indépendante du bloc d'intervention. Les valeurs ci-dessous sont les augmentations de NLL provoquées par le retrait de la base linguistique ; une valeur faible préserve mieux le comportement du modèle.

| Modèle | Bloc | EN | KO | JA | ZH | Moyenne |
|---|---:|---:|---:|---:|---:|---:|
| Tri | 7 | +1,1935 | +7,0332 | +5,0586 | +2,4778 | +3,9407 |
| Tri | 22 | +1,1947 | +1,0234 | +1,3635 | +3,5291 | +1,7777 |
| Qwen | 7 | +2,6919 | +4,6388 | +6,3836 | +8,6279 | +5,5855 |
| Qwen | 22 | +0,2069 | +3,3088 | +2,6698 | +2,3163 | +2,1255 |

Les interventions tardives sont nettement moins destructrices en moyenne. Le bloc 7 de Qwen produit le meilleur transfert MASSIVE de Qwen, mais aussi le coût LM le plus élevé des quatre configurations. Il ne peut donc pas être qualifié de meilleur point d'intervention sans préciser le compromis recherché.

## Interprétation révisée

1. **La transition mécanistique de Tri au bloc 7 est spécifique.** Le MLP du bloc 7 fait chuter fortement le retrieval en mean pooling tout en faisant passer la fraction de variance linguistique de 0,229 à 0,983. Rien de comparable ne se produit au bloc 22 de Tri.
2. **L'effet causal de Tri n'est toutefois pas localisé uniquement au bloc 7.** En mean pooling, intervenir au bloc 22 améliore même légèrement plus le résultat final qu'au bloc 7. En last-token pooling, cet effet tardif disparaît. Il faut donc séparer la localisation du changement géométrique brutal de la localisation optimale d'une intervention.
3. **Chez Qwen, le bloc 22 reste nettement plus déterminant pour FLORES, mais pas pour MASSIVE.** Le gain final FLORES au bloc 22 est environ 4,4 fois supérieur au bloc 7 en mean pooling et 3,2 fois supérieur en last-token pooling. Sur MASSIVE, le bloc 7 donne au contraire les meilleurs gains, ce qui montre que la localisation causale dépend de la tâche mesurée.
4. **Le signal précoce de Qwen n'est pas spécifiquement linguistique sur FLORES, mais il transfère de façon non triviale.** Au bloc 7, la PCA égale ou dépasse la base linguistique sur FLORES. Sur MASSIVE en mean pooling, la base linguistique dépasse toutefois la PCA de +0,0604, IC 95 % [0,0460 ; 0,0758]. Une interprétation purement « directions dominantes générales » serait donc trop forte.
5. **Le coût LM exclut de présenter les interventions comme une amélioration gratuite.** Les gains MASSIVE du bloc 7 de Qwen s'accompagnent d'une hausse moyenne de NLL de +5,5855, contre +2,1255 au bloc 22. Chez Tri, le bloc 22 offre le meilleur compromis en mean pooling et réduit le coût moyen de +3,9407 à +1,7777.
6. **La conclusion inter-modèles est renforcée mais doit être conditionnelle à la métrique et au pooling.** Tri présente une transition géométrique précoce et abrupte au bloc 7. Qwen a son plus grand levier FLORES dans les blocs tardifs, tandis que son meilleur gain MASSIVE apparaît plus tôt. La couche où naît une géométrie, la couche optimale pour une tâche et la couche la moins dommageable pour le LM ne sont pas nécessairement les mêmes.

## Sorties

- `raw/qwen_05b_block_07_retrieval.csv`
- `raw/qwen_05b_block_07_mechanism.csv`
- `raw/qwen_05b_intervention_basis_comparison.csv`
- `raw/tri_05b_block_22_retrieval.csv`
- `raw/tri_05b_block_22_mechanism.csv`
- `raw/tri_05b_intervention_basis_comparison.csv`
- `raw/qwen_05b_massive_intervention.csv`
- `raw/tri_05b_massive_intervention.csv`
- `raw/qwen_05b_intervention_lm_eval.csv`
- `raw/tri_05b_intervention_lm_eval.csv`
- les sept figures correspondantes dans `figures/`, dont la synthèse MASSIVE combinée
