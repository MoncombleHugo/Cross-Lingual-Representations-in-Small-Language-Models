# Cross-Lingual Representations in Small Language Models

This project explores how multilingual language models organize meaning across languages, and how that organization changes through the network.

The main question is simple: **when two sentences mean the same thing but are written in different languages, how similarly are they represented inside the model?**

I study this in two small causal language models, **Tri-0.5B** and **Qwen2.5-0.5B**, using English, Korean, Japanese, and Chinese.

## What the project investigates

The first step is to track cross-lingual alignment layer by layer. Parallel sentences become increasingly easy to match in intermediate layers, but this alignment can deteriorate sharply near the end of the model.

This raises an important distinction: does the model actually lose shared semantic information, or does that information simply become harder to recover from the raw geometry?

To investigate this, the project combines:

- cross-lingual retrieval across layers
- linear probes for language identity
- language-wise centering and alignment
- low-rank interventions in the residual stream
- PCA and random-direction controls
- zero-shot transfer on MASSIVE
- next-token likelihood evaluation

Rather than treating any single metric as evidence of a mechanism, the experiments progressively test alternative explanations for the observed geometry.

## What we learn

The results suggest that **cross-lingual alignment and language identity are not opposites**. Representations can remain strongly language-specific while still supporting good semantic alignment across languages.

The degradation of cross-lingual retrieval in late layers also does not necessarily mean that shared semantic structure has disappeared. Much of it can be recovered by removing simple language-dependent offsets or a very small number of dominant directions.

Residual-stream interventions show that these low-dimensional directions can actively influence the geometry produced by later layers. However, the controls are important: in Qwen, much of the effect can also be reproduced by removing dominant PCA directions, while in Tri the language-derived subspace appears more distinct from this generic high-variance structure.

This means that strong intervention effects should not automatically be interpreted as evidence for a uniquely language-specific mechanism.

Finally, making representations more cross-lingually aligned does **not** necessarily make the language model better. The same interventions that improve retrieval and zero-shot transfer also worsen next-token prediction. The directions that make the geometry less convenient for cross-lingual comparison can still be useful for the model's actual objective.

## Takeaway

The project points toward a view where multilingual representations are not progressively compressed into a single language-independent semantic space.

Instead, shared semantic structure, language-specific information, and high-variance directions coexist and are reorganized across layers. Intermediate representations can be especially easy to compare across languages, while later layers reshape that geometry for next-token prediction.

More broadly, the experiments illustrate why representation analysis benefits from combining geometric measurements, causal interventions, and matched controls before drawing conclusions about what a model is actually doing.

## Repository structure

```text
configs/                              Experiment configurations
src/cross_lingual_representations/   Analysis code
scripts/                              Experiment entry points
results/                              Tables and figures
tests/                                Tests
