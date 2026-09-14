# Cross-Lingual Representations in Small Language Models

This project studies **cross-lingual representation geometry in small causal language models**, focusing on how semantic alignment evolves across layers and how low-dimensional residual-stream interventions affect that geometry.

The experiments compare **Tri-0.5B** and **Qwen2.5-0.5B** on English, Korean, Japanese, and Chinese using **FLORES-200** and **MASSIVE**.

## Overview

The analysis combines:

- layer-wise cross-lingual retrieval
- linear language probes
- orthogonal Procrustes alignment
- low-rank residual-stream interventions
- matched PCA and random controls
- zero-shot transfer evaluation
- next-token NLL measurements

For the intervention experiments, a rank-3 language-associated subspace is estimated from centered language centroids and removed from an intermediate residual stream:

h' = h - UU^T h

The modified hidden state is then propagated through the remaining Transformer blocks.

## Main results

Both models show that a very small number of directions can strongly affect final-layer cross-lingual geometry.

| Model | FLORES R@1 | After intervention |
|---|---:|---:|
| Tri-0.5B | 0.124 | **0.325** |
| Qwen2.5-0.5B | 0.043 | **0.706** |

The same interventions also improve zero-shot performance on **MASSIVE**, indicating that the effect is not limited to translation retrieval.

The control experiments reveal different behaviors across models:

- **Qwen:** PCA removal reproduces much of the intervention gain, suggesting that dominant high-variance directions explain a large part of the effect.
- **Tri:** the language-derived basis separates more clearly from the PCA control, indicating a stronger language-associated component.

The interventions also increase next-token NLL, showing that directions that hurt cross-lingual geometry can still be useful for language modeling.

## Takeaway

The results show that **cross-lingual geometry in late layers can be highly sensitive to low-dimensional structure**. They also highlight the importance of matched controls when interpreting intervention effects as language-specific.

## Repository structure

configs/                              Experiment configurations
src/cross_lingual_representations/   Analysis code
scripts/                              Experiment entry points
results/                              Tables and figures
tests/                                Tests

Run the main pipelines with:

python scripts/run_all.py --config configs/tri_05b_public.yaml
python scripts/run_all.py --config configs/qwen_05b_public.yaml

## Scope

Current experiments cover two 0.5B models, four languages, FLORES-200, and MASSIVE. The Tri/Qwen comparison should be interpreted with some caution because the selected intervention layers occur at different relative depths.
