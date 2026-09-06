# Cross-Lingual Representations in Small Language Models

**A layer-wise study of how multilingual representations emerge in sub-billion-parameter decoder-only language models**

---

## Implementation status

A self-contained French walkthrough of the methodology, experiments, figures, interpretation, and
technical caveats is available in
[`docs/RAPPORT_EXPERIMENTAL.md`](docs/RAPPORT_EXPERIMENTAL.md).

Phases 1–12 of the implementation plan are complete:

- an installable `src/`-layout package, configurations, scripts, tests, and ignored output directories;
- deterministic FLORES `dev`/`devtest` loading from the official `facebook/flores` `all`
  configuration, with one shared row sample and explicit ID/column validation;
- generic Hugging Face causal-LM loading with CPU/CUDA and dtype resolution, right padding,
  disabled KV caching, and a one-sentence hidden-state inspection command.
- masked mean and final non-padding-token pooling applied immediately after each forward pass;
- validated `.npz` sentence-representation caches with explicit compatibility checks;
- cosine translation retrieval with ID-aware ranking, R@1/R@5/R@10/MRR, matched/unmatched
  controls, and average-rank handling for exact ties;
- evaluation of every directed pair among EN/KO/JA/ZH, tidy CSV output, and headless PNG figures
  generated from the saved result table.
- orthogonal Procrustes fitted on `dev` only and evaluated on `devtest`, with raw/aligned/delta
  retrieval metrics and a synthetic rotation-orientation test; degenerate zero-norm centered layers
  are retained as explicit `NaN` values rather than receiving fabricated scores;
- standardized logistic-regression language probes with accuracy, macro F1, and confusion matrices
  for the embedding, middle, and final states;
- sentence-level tokenizer measurements (characters, UTF-8 bytes, tokens, normalized rates, and
  aligned ratios to English) with multi-model summaries and figures;
- a configuration-only Qwen2.5-0.5B baseline run through the same extraction and evaluation code;
- consolidated final tables, cross-model figures, best-layer heatmaps, and an evidence-based
  findings section generated from the saved tidy CSV files;
- a restartable `run_all.py` orchestrator, isolated smoke outputs, environment diagnostics,
  an execution manifest, and a successful end-to-end smoke validation;
- train-only per-language centering separated from Procrustes rotation, 1,000-query bootstrap
  intervals, 20 query-subsample stability runs, anisotropy and stable-rank diagnostics,
  relative-token-position retrieval, and targeted residual-stream decompositions of Tri block 7
  and Qwen block 22.

The reported final run used the official gated `facebook/flores` repository after authenticating
locally and accepting its access conditions. The explicit `*_public.yaml` configurations remain as
an unauthenticated fallback for development, but they are not the source of the reported results.
No token is read from a project file or committed to Git.

### Local GPU and Hugging Face setup

The final experiments ran on an NVIDIA GeForce RTX 3080 with PyTorch `2.11.0+cu128`; CUDA detection
was recorded in each pipeline manifest. If a fresh environment installs a CPU-only PyTorch build,
install a CUDA wheel using the command recommended by the
[official PyTorch selector](https://pytorch.org/get-started/locally/). For CUDA 12.8:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Authentication is unnecessary for the public fallback and the two model repositories. It is
required for the official FLORES configurations: log in on Hugging Face, accept the conditions on
[`facebook/flores`](https://huggingface.co/datasets/facebook/flores), then run:

```powershell
.\.venv\Scripts\hf.exe auth login
```

`run_all.py` checks both conditions up front: it warns when an NVIDIA GPU is paired with CPU-only
PyTorch and stops with an actionable message if an official gated configuration has no local token.

```bash
python -m venv .venv
pip install -e ".[dev]"
python scripts/inspect_model.py --model trillionlabs/Tri-0.5B-Base
for config in configs/tri_05b.yaml configs/qwen_05b.yaml; do
  python scripts/extract_representations.py --config "$config" --split train
  python scripts/extract_representations.py --config "$config" --split evaluation
  python scripts/run_retrieval.py --config "$config"
  python scripts/run_procrustes.py --config "$config"
  python scripts/run_language_probe.py --config "$config"
done
python scripts/run_tokenization_analysis.py --config configs/tri_05b.yaml --config configs/qwen_05b.yaml
python scripts/make_final_analysis.py --config configs/tri_05b.yaml --config configs/qwen_05b.yaml
pytest
ruff check .
mypy src scripts
```

The complete smoke pipeline is a single command. Its caches and results are isolated under
`artifacts/smoke/` and `results/smoke/`, so it cannot overwrite the final experiment outputs:

```bash
python scripts/run_all.py --config configs/smoke.yaml
```

The phase-3 inspection was verified on CPU with Tri-0.5B-Base. For the sentence
`The train arrives at six.`, the model reported 471,870,336 parameters, 25 hidden-state tensors
(embedding output plus 24 Transformer blocks), hidden size 896, and state shapes
`[1, 6, 896]`. These are observed implementation checks, not experimental findings.

---

## 1. Project Overview

Modern multilingual language models are trained on text spanning many languages, but it is not obvious **how representations of equivalent meaning become aligned across languages inside the network**.

Consider two sentences expressing the same idea:

> English: “The train arrives at six.”  
> Korean: “기차는 6시에 도착합니다.”

A multilingual model processes these sentences as very different sequences of tokens. However, if it has learned transferable multilingual representations, internal representations of these sentences may become increasingly similar as information passes through the Transformer.

This project studies that phenomenon directly.

The central research question is:

> **At which Transformer layers do semantically equivalent sentences written in different languages become aligned in small multilingual causal language models?**

Rather than training another language model, the project treats existing pretrained models as objects of study. It extracts sentence-level representations from every layer and evaluates their structure using retrieval, geometric alignment, lightweight probing, and tokenization analysis.

The main model of interest is:

```text
trillionlabs/Tri-0.5B-Base
```

A similarly sized multilingual base model is used as a comparison:

```text
Qwen/Qwen2.5-0.5B
```

The project focuses on the four languages explicitly supported by Tri-0.5B:

```text
English
Korean
Japanese
Simplified Chinese
```

The project is deliberately designed to be:

- scientifically interpretable;
- computationally lightweight;
- reproducible;
- modular;
- easy to extend;
- implementable in approximately three days;
- representative of real ML research engineering rather than notebook-only experimentation.

---

# 2. Motivation

## 2.1 Why study internal representations?

A multilingual model may produce reasonable text in several languages without necessarily organizing those languages identically internally.

Several possibilities could occur.

The model could maintain mostly language-specific representations throughout the network.

Alternatively, early layers could encode language-specific lexical and syntactic information while later layers map sentences toward a more language-independent semantic representation.

Another possibility is that multilingual representations remain separated but have similar geometric structure, meaning that a simple transformation can map one language space onto another.

These possibilities can be investigated without training a large model.

By examining hidden states across Transformer depth, we can study questions such as:

- When does cross-lingual semantic alignment emerge?
- Is alignment gradual or concentrated in specific layers?
- Do different language pairs behave differently?
- Are representations directly aligned or merely structurally similar?
- Does the model retain information about language identity while learning cross-lingual semantics?
- How much of the observed behavior can be explained by differences in tokenization?

---

## 2.2 Why small language models?

This project intentionally focuses on models around 500M parameters.

The goal is not to demonstrate access to large amounts of compute. The goal is to demonstrate the ability to formulate a useful research question, design controlled experiments, implement them cleanly, and extract meaningful conclusions with limited resources.

Small models make it possible to inspect every layer across hundreds of multilingual examples on modest hardware.

No model training is required for the core project.

The only learned components are:

- tiny linear classifiers;
- closed-form orthogonal transformations.

These operations are negligible compared with LLM training.

---

# 3. Relationship to Multilingual Foundation Model Research

This project is intentionally **adjacent to multilingual foundation-model development rather than a reproduction of a particular training method**.

It does not attempt to:

- reproduce Trillion Labs' pretraining;
- reproduce their cross-lingual training techniques;
- train a Korean foundation model;
- reproduce tokenizer research;
- compare models on a large benchmark leaderboard;
- fine-tune Tri-0.5B to solve another downstream task.

Instead, it asks an interpretability/evaluation question:

> **What does cross-lingual alignment look like inside a multilingual causal language model?**

This makes the project relevant to multilingual language-model research while remaining independently motivated.

---

# 4. Research Questions

The project should answer four primary questions.

## RQ1 — Layer-wise semantic alignment

**How does cross-lingual semantic retrieval performance evolve through Transformer depth?**

Given a sentence in language A, can its translation in language B be retrieved using the hidden representation of the sentence?

We evaluate this independently at every layer.

---

## RQ2 — Geometric alignment

**Are multilingual representations already similar up to a simple geometric transformation?**

Even when raw vectors from two languages are not perfectly aligned, their representation spaces may have similar geometry.

We test this by learning an orthogonal Procrustes transformation between languages using a small training set of parallel sentences.

---

## RQ3 — Language information

**Does increasing semantic alignment imply that representations stop encoding language identity?**

We train a simple linear classifier to predict the language from the representation at each layer.

This lets us compare:

```text
semantic alignment
        versus
language separability
```

A representation can potentially become useful for cross-lingual retrieval while still retaining strong information about language identity.

---

## RQ4 — Tokenization

**How differently are equivalent multilingual sentences represented by the models' tokenizers?**

We measure sequence length and tokenization efficiency for each language and model.

This provides context for interpreting representation-level results.

Tokenization is a secondary analysis, not the main research question.

---

# 5. Hypotheses

These hypotheses guide the analysis but **must never be treated as expected results that the implementation should try to confirm**.

### H1 — Alignment changes with depth

Cross-lingual retrieval quality will vary substantially across Transformer layers rather than remaining constant.

### H2 — Alignment may emerge most strongly after early layers

Early representations are expected to contain substantial lexical and language-specific information. Intermediate or later layers may contain stronger cross-lingual semantic structure.

This is a hypothesis only.

### H3 — Language pairs will behave differently

English–Korean, English–Japanese, and English–Chinese may exhibit different alignment dynamics.

### H4 — Orthogonal alignment may improve retrieval

If representations across languages have similar geometry but different orientations, Procrustes alignment should increase cross-lingual retrieval performance.

### H5 — Semantic alignment and language identity can coexist

High translation retrieval performance does not necessarily imply that language information disappears.

### H6 — Models may exhibit different alignment dynamics

Tri-0.5B and the baseline model may show different layer-wise behavior.

There is **no hypothesis that one model must outperform the other**.

Negative, mixed, or unexpected results are valid research outcomes.

---

# 6. Models

## 6.1 Primary model

```text
trillionlabs/Tri-0.5B-Base
```

Relevant properties:

```text
Architecture: decoder-only Transformer
Approximate parameters: 472M
Transformer layers: 24
Hidden dimension: 896
Languages of interest:
    English
    Korean
    Japanese
    Chinese
Training stage:
    base pretraining
```

This is the main model studied by the project.

---

## 6.2 Comparison model

```text
Qwen/Qwen2.5-0.5B
```

Use the **base model**, not the instruction-tuned version.

Relevant properties:

```text
Architecture: decoder-only Transformer
Approximate parameters: 0.49B
Transformer layers: 24
Languages include:
    English
    Korean
    Japanese
    Chinese
Training stage:
    base pretraining
```

The similar parameter count and depth make this a useful comparison while remaining computationally inexpensive.

---

## 6.3 Important comparison rule

The objective is **not to establish a leaderboard between Tri and Qwen**.

Differences in:

- training corpora;
- tokenizers;
- objectives;
- model architecture;
- vocabulary;
- data mixture;

make strong causal claims inappropriate.

The comparison should therefore be framed as:

> “Do two similarly sized multilingual causal LMs exhibit similar layer-wise alignment behavior?”

rather than:

> “Which model is better?”

---

# 7. Dataset

Use **FLORES-200** parallel sentences.

Relevant language identifiers:

```yaml
en: eng_Latn
ko: kor_Hang
ja: jpn_Jpan
zh: zho_Hans
```

FLORES is useful because the same semantic content is professionally translated into multiple languages.

For sentence index `i`:

```text
English[i]
Korean[i]
Japanese[i]
Chinese[i]
```

should express equivalent content.

This gives us direct cross-lingual positive pairs without generating translations ourselves.

---

# 8. Experimental Splits

Data leakage must be avoided.

Use separate FLORES splits for fitting lightweight transformations and final evaluation.

Recommended configuration:

```yaml
train_split: dev
evaluation_split: devtest

n_train: 256
n_eval: 512

seed: 42
```

If compute is limited:

```yaml
n_train: 128
n_eval: 256
```

For smoke testing:

```yaml
n_train: 16
n_eval: 32
```

The same sentence indices must be selected for all languages.

Never independently sample examples for different languages.

---

# 9. Compute Constraints

The project must be designed for limited hardware.

## Requirements

The core experiments must:

- require no LLM training;
- run models sequentially rather than simultaneously;
- support CUDA if available;
- support CPU fallback;
- avoid storing full token-level hidden states;
- avoid computing or storing attention matrices;
- disable generation;
- disable KV caching;
- operate under `torch.inference_mode()`.

Do not introduce distributed training, Kubernetes, DeepSpeed, FSDP, or other unnecessary infrastructure.

This project is about extracting insight efficiently, not demonstrating infrastructure complexity.

---

# 10. Core Experimental Pipeline

The complete pipeline is:

```text
                 FLORES parallel sentences
                          │
                          ▼
                    Data loader
                          │
                  ┌───────┴───────┐
                  ▼               ▼
              Tri-0.5B       Qwen2.5-0.5B
                  │               │
                  ▼               ▼
             tokenization     tokenization
                  │               │
                  ▼               ▼
        layer-wise hidden representations
                  │
                  ▼
             sentence pooling
                  │
           ┌──────┼───────────────┐
           ▼      ▼               ▼
       Retrieval  Procrustes    Language probe
           │      alignment        │
           └──────┼────────────────┘
                  ▼
              metrics tables
                  │
                  ▼
               figures
                  │
                  ▼
         interpretation / README
```

---

# 11. Representation Extraction

## 11.1 Hidden states

Models should be called with:

```python
output_hidden_states=True
use_cache=False
```

and inference should run under:

```python
torch.inference_mode()
```

The implementation must not assume a fixed number of returned hidden-state tensors.

Instead:

```python
hidden_states = outputs.hidden_states
num_states = len(hidden_states)
```

Record this value in experiment metadata.

For Hugging Face causal language models, this will typically include the embedding output plus one representation per Transformer block.

Use labels such as:

```text
layer_00_embedding
layer_01
layer_02
...
layer_24
```

but derive them programmatically.

Do not silently assume that hidden-state conventions are identical across model implementations.

---

# 12. Sentence Representations

Turning token-level hidden states into sentence-level vectors is a key methodological choice.

Because these are **causal decoder models**, token `t` only has access to tokens at positions `<= t`.

Therefore, naive mean pooling has a subtle disadvantage: many token states represent only sentence prefixes.

For this reason, the **primary representation should be the last non-padding token**.

---

## 12.1 Primary pooling method — last token

For each layer:

```python
length = attention_mask.sum(dim=1)
index = length - 1
sentence_vector = hidden_state[batch_index, index]
```

This token has access to the full preceding sentence under causal attention.

Set:

```python
tokenizer.padding_side = "right"
```

for extraction.

If the tokenizer has no padding token, use its EOS token as padding where appropriate, but do not modify the model vocabulary.

---

## 12.2 Secondary pooling method — masked mean

As a robustness check:

\[
z_l(x)
=
\frac{
\sum_t m_t h_{l,t}
}{
\sum_t m_t
}
\]

where `m_t` is the attention mask.

This must ignore padding tokens.

---

## 12.3 Required pooling methods

Implement:

```text
last_token
mean
```

Do not add additional pooling strategies unless the core project is complete.

---

# 13. Representation Normalization

Cosine retrieval requires L2-normalized vectors.

Given representation \(z\):

\[
\hat z = \frac{z}{\|z\|_2}
\]

Normalization should happen immediately before similarity computation.

Keep raw cached sentence representations unchanged.

This allows different post-processing strategies to be tested without repeating model inference.

---

# 14. Experiment 1 — Layer-Wise Cross-Lingual Retrieval

This is the central experiment.

Suppose there are `N` parallel English and Korean sentences.

At layer `l`, build matrices:

\[
X_l \in \mathbb{R}^{N\times d}
\]

for English and:

\[
Y_l \in \mathbb{R}^{N\times d}
\]

for Korean.

Row `i` in `X` and row `i` in `Y` correspond to translations.

Normalize all rows.

Compute:

\[
S = X Y^T
\]

where:

\[
S_{ij} = \cos(X_i,Y_j)
\]

For each source sentence `i`, rank all target-language sentences according to `S[i]`.

The correct answer is target `i`.

---

# 15. Retrieval Metrics

Compute the following.

## Recall@1

Fraction of queries whose correct translation is ranked first.

\[
R@1 =
\frac{1}{N}
\sum_i
\mathbf{1}[\operatorname{rank}_i = 1]
\]

---

## Recall@5

\[
R@5 =
\frac{1}{N}
\sum_i
\mathbf{1}[\operatorname{rank}_i \leq 5]
\]

---

## Recall@10

Optional but cheap.

---

## Mean Reciprocal Rank

\[
MRR =
\frac{1}{N}
\sum_i
\frac{1}{\operatorname{rank}_i}
\]

---

# 16. Retrieval Directions

At minimum evaluate:

```text
EN → KO
KO → EN

EN → JA
JA → EN

EN → ZH
ZH → EN
```

If compute and analysis time permit, evaluate all 12 directed pairs:

```text
EN ↔ KO
EN ↔ JA
EN ↔ ZH
KO ↔ JA
KO ↔ ZH
JA ↔ ZH
```

Since hidden states are already cached, extending to every pair has negligible computational cost.

Therefore the preferred final experiment uses **all directed pairs**.

---

# 17. Retrieval Controls

Include simple sanity checks.

For `N` candidates, random Recall@1 is approximately:

\[
1/N
\]

and random Recall@5 approximately:

\[
5/N
\]

Report these baselines in plots where useful.

Also compute the distributions of cosine similarities for:

```text
matched translation pairs
random unmatched pairs
```

This provides a useful sanity check beyond ranking metrics.

---

# 18. Primary Retrieval Figure

Generate one figure per model:

```text
translation retrieval R@1
        vs
normalized Transformer depth
```

Include one line for each major language direction or use pair-level averages if the plot becomes unreadable.

Also generate a compact comparison figure showing average retrieval across all evaluated language directions.

Because both initial models have 24 Transformer layers, raw layer index is easy to compare.

Nevertheless, the plotting code should also expose:

\[
\text{normalized depth}
=
\frac{\text{layer index}}
{\text{number of layers}}
\]

so future models with different depths can be compared.

---

# 19. Experiment 2 — Orthogonal Procrustes Alignment

Raw representations may not occupy exactly the same coordinate system across languages.

However, their geometry may still be similar.

We test this using **Orthogonal Procrustes alignment**.

Given paired training representations:

\[
X \in \mathbb{R}^{N\times d},
\qquad
Y \in \mathbb{R}^{N\times d}
\]

learn an orthogonal matrix:

\[
W^* =
\arg\min_{W^TW=I}
\|XW-Y\|_F^2
\]

The solution is obtained efficiently using SVD.

If:

\[
X^TY = U\Sigma V^T
\]

then:

\[
W = UV^T
\]

depending on the exact SVD convention used in the implementation.

Unit tests must verify the orientation rather than relying on memory.

---

# 20. Procrustes Data Protocol

**Never fit and evaluate Procrustes on the same sentences.**

Use:

```text
FLORES dev
    → fit alignment

FLORES devtest
    → evaluate retrieval
```

For each:

```text
model
layer
language pair
pooling method
```

perform the following.

### Step 1

Extract training matrices:

```text
X_train
Y_train
```

### Step 2

Compute training-set means:

```text
mu_X
mu_Y
```

### Step 3

Center:

```text
Xc = X_train - mu_X
Yc = Y_train - mu_Y
```

### Step 4

Fit orthogonal Procrustes:

```text
W
```

### Step 5

For evaluation data:

```text
X_test_aligned = (X_test - mu_X) @ W
Y_test_centered = Y_test - mu_Y
```

### Step 6

L2-normalize both spaces.

### Step 7

Run the exact same retrieval evaluation as the raw representation experiment.

---

# 21. Procrustes Output

For each layer report:

```text
raw R@1
aligned R@1
delta R@1

raw MRR
aligned MRR
delta MRR
```

The important quantity is often:

```text
alignment gain = aligned metric - raw metric
```

---

# 22. Interpretation of Procrustes

If Procrustes strongly improves retrieval, a reasonable interpretation is:

> The two language representation spaces contain related geometric structure even when the original coordinates are imperfectly aligned.

Do **not** claim:

> The model has learned a universal language-independent semantic space.

The experiment does not establish that.

---

# 23. Experiment 3 — Linear Language Probe

We next ask:

> How easy is it to determine the language from representations at each layer?

For each layer, collect sentence vectors from:

```text
English
Korean
Japanese
Chinese
```

The label is the language.

Use a linear classifier:

```python
sklearn.linear_model.LogisticRegression
```

Do not train neural probes.

A simple linear probe is deliberately chosen because we want to measure **linearly accessible information**, not the capacity of a downstream classifier.

---

# 24. Language Probe Protocol

Training data:

```text
FLORES dev
```

Evaluation data:

```text
FLORES devtest
```

Because every sentence exists in every language, semantic content is approximately balanced across language classes.

This is useful: the classifier cannot simply associate different topics with different languages.

Pipeline:

```python
StandardScaler()
LogisticRegression(...)
```

Recommended settings:

```text
C = 1.0
max_iter = 2000
random_state = 42
multi-class classification
```

Avoid hyperparameter searches.

The goal is probing, not optimizing classification accuracy.

---

# 25. Probe Metrics

Report:

```text
accuracy
macro F1
```

Because there are four balanced classes, random accuracy is approximately:

```text
25%
```

A confusion matrix may be generated for a small number of representative layers:

```text
embedding / first layer
middle layer
final layer
```

Do not generate 25 confusion matrices.

---

# 26. Combined Alignment / Language Figure

One of the most interesting final plots should display, for each layer:

```text
Cross-lingual retrieval score
Language classification accuracy
```

These may be shown as two separate aligned plots sharing the same x-axis.

Avoid a misleading dual-y-axis visualization unless clearly labeled.

The analysis should ask:

- Does retrieval improve while language classification stays high?
- Do both decrease?
- Do they peak at different layers?
- Are there transition regions?

Do not infer causality.

---

# 27. Experiment 4 — Tokenization Analysis

Tokenization is a supporting analysis.

For every:

```text
model
language
sentence
```

record:

```text
Unicode character count
UTF-8 byte count
token count
```

From these values compute:

```text
tokens per sentence
tokens per 100 characters
bytes per token
characters per token
```

Also compute sequence-length ratios relative to English for aligned sentences.

For example:

\[
r_{\mathrm{KO/EN}}
=
\frac{
\text{tokens(Korean translation)}
}{
\text{tokens(English sentence)}
}
\]

---

# 28. Tokenization Caveat

Do not interpret token count differences purely as tokenizer quality.

Translations naturally differ in:

- number of characters;
- morphology;
- word boundaries;
- syntax;
- information density.

For this reason, report several normalization measures rather than only raw token counts.

Comparisons **between tokenizers for the same language and same sentences** are generally easier to interpret than comparisons of raw token counts across languages.

---

# 29. Optional Robustness Analysis — Centered Representations

Raw Transformer representations may be anisotropic.

Cosine similarity can therefore be influenced by a large common component shared across many sentences.

After the core experiments work, implement an optional centered representation condition:

```text
raw
centered
```

For centered retrieval:

```python
mean = train_embeddings.mean(axis=0)
test_centered = test_embeddings - mean
test_centered = l2_normalize(test_centered)
```

For multilingual retrieval, decide explicitly whether centering is:

```text
global
```

or:

```text
per-language
```

The recommended robustness test is **per-language centering using only training data**.

Never compute centering statistics on evaluation data.

This experiment is optional and should not delay the core deliverable.

It is now implemented as an explicit `centered` condition alongside `raw` and fully `aligned`
retrieval. Language means are fitted on `dev`; the results show that centering alone explains most
of the late-layer recovery.

---

# 30. Repository Structure

Use the following structure.

```text
cross-lingual-representations/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock                       # or another reproducible lockfile
├── .gitignore
│
├── configs/
│   ├── smoke.yaml
│   ├── tri_05b.yaml
│   ├── qwen_05b.yaml
│   └── full_experiment.yaml
│
├── src/
│   └── crosslingual/
│       ├── __init__.py
│       │
│       ├── config.py
│       ├── data.py
│       ├── models.py
│       ├── extraction.py
│       ├── pooling.py
│       ├── cache.py
│       │
│       ├── similarity.py
│       ├── retrieval.py
│       ├── procrustes.py
│       ├── probes.py
│       ├── tokenization.py
│       ├── metrics.py
│       │
│       ├── results.py
│       └── plotting.py
│
├── scripts/
│   ├── extract_representations.py
│   ├── run_retrieval.py
│   ├── run_procrustes.py
│   ├── run_language_probe.py
│   ├── run_tokenization_analysis.py
│   ├── make_figures.py
│   └── run_all.py
│
├── tests/
│   ├── test_data.py
│   ├── test_pooling.py
│   ├── test_retrieval.py
│   ├── test_procrustes.py
│   └── test_metrics.py
│
├── data/
│   └── .gitkeep
│
├── artifacts/
│   ├── representations/
│   └── metadata/
│
├── results/
│   ├── raw/
│   ├── tables/
│   └── figures/
│
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

# 31. Repository Philosophy

The notebook is not the project.

Core functionality must live in:

```text
src/crosslingual/
```

Scripts should only:

1. parse configuration;
2. call reusable library functions;
3. save outputs.

The notebook may be used for exploratory visualization, but every final experiment and figure must be reproducible without executing the notebook.

---

# 32. Configuration

Experiments should be controlled from YAML files.

Example:

```yaml
experiment_name: tri_main

model:
  name: trillionlabs/Tri-0.5B-Base
  dtype: auto
  device: auto

data:
  dataset: flores200

  languages:
    en: eng_Latn
    ko: kor_Hang
    ja: jpn_Jpan
    zh: zho_Hans

  train_split: dev
  eval_split: devtest

  n_train: 256
  n_eval: 512

  seed: 42

extraction:
  batch_size: 8
  max_length: 256
  pooling:
    - last_token
    - mean

retrieval:
  metrics:
    - recall_at_1
    - recall_at_5
    - recall_at_10
    - mrr

procrustes:
  enabled: true

language_probe:
  enabled: true

tokenization:
  enabled: true

output:
  representation_dtype: float16
```

Avoid hardcoding experiment values inside scripts.

---

# 33. Model Loading

Implement a generic model loader.

Pseudo-interface:

```python
@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    model_name: str
    num_layers: int
    hidden_size: int
```

Loader requirements:

```python
load_model(model_name, device, dtype)
```

must:

- load tokenizer;
- load causal LM;
- set evaluation mode;
- infer architecture metadata;
- configure a valid padding token;
- set right padding;
- never enable text generation;
- never modify pretrained weights.

Use:

```python
model.eval()
```

and:

```python
torch.inference_mode()
```

during extraction.

---

# 34. Device Handling

Support:

```text
CUDA
MPS
CPU
```

Preferred behavior:

```text
device=auto
```

should select:

```text
CUDA if available
MPS if available
CPU otherwise
```

Precision:

```text
CUDA:
    prefer bfloat16 when supported
    otherwise float16

MPS:
    float16 when appropriate

CPU:
    float32 unless a tested lower-precision path works reliably
```

Do not force BF16 on unsupported hardware.

---

# 35. Memory-Efficient Extraction

**Do not save token-level hidden states to disk.**

For each batch:

```text
tokens
   ↓
model
   ↓
hidden states for all layers
   ↓
pool each layer immediately
   ↓
move pooled vectors to CPU
   ↓
discard token-level states
```

Only sentence-level vectors should be cached.

For one sentence:

```text
num_layers × hidden_dimension
```

rather than:

```text
num_layers × sequence_length × hidden_dimension
```

This dramatically reduces storage.

---

# 36. Representation Cache

Model inference is the expensive part.

Every extracted representation must be cached.

Suggested path:

```text
artifacts/representations/
    trillionlabs__Tri-0.5B-Base/
        dev/
            last_token.npz
            mean.npz
        devtest/
            last_token.npz
            mean.npz
```

The cache must include:

```text
sentence IDs
languages
layer indices
representation vectors
model name
pooling strategy
dataset split
seed
max sequence length
dtype
timestamp or experiment identifier
```

Prefer `.npz`, `.npy`, or another simple non-proprietary format.

Do not pickle arbitrary Python objects.

---

# 37. Cache Validation

Before reusing cached representations, verify that metadata matches the requested experiment.

At minimum check:

```text
model_name
split
languages
indices
pooling
max_length
```

If incompatible, fail clearly rather than silently reusing stale results.

---

# 38. Result Format

All experiment outputs should ultimately be written as tidy tabular files.

Example retrieval CSV:

```text
model,layer,normalized_depth,pooling,source_language,target_language,condition,metric,value,n
Tri-0.5B,0,0.00,last_token,en,ko,raw,r1,0.123,512
Tri-0.5B,1,0.04,last_token,en,ko,raw,r1,0.145,512
...
```

Example probe CSV:

```text
model,layer,normalized_depth,pooling,metric,value
Tri-0.5B,0,0.00,last_token,accuracy,0.96
...
```

Example tokenization CSV:

```text
model,language,sentence_id,characters,bytes,tokens
...
```

Figures must be generated from these result files.

Never manually type result values into plotting code.

---

# 39. Reproducibility

Implement a global seed utility controlling at minimum:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

When CUDA is available:

```python
torch.cuda.manual_seed_all(seed)
```

The selected FLORES indices should be saved explicitly.

For example:

```text
artifacts/metadata/train_indices.json
artifacts/metadata/eval_indices.json
```

This ensures a future rerun evaluates exactly the same examples.

---

# 40. Statistical Uncertainty

If time permits, use bootstrap confidence intervals for the main retrieval metric.

Recommended:

```text
1000 bootstrap resamples
95% percentile confidence interval
```

Resample queries, not individual similarity matrix entries.

This is computationally cheap because representations are already extracted.

Bootstrap intervals are a **nice-to-have**, not required before the basic experiments work.

They are now reported using 1,000 joint query-ID resamples with 95% percentile intervals; the same
resampled IDs are shared across all 12 directions.

---

# 41. Required Figures

The final repository should contain at least the following.

## Figure 1 — Layer-wise retrieval

```text
Cross-lingual Recall@1 vs Transformer layer
```

Main figure.

---

## Figure 2 — Language-pair heatmap

For one or several representative layers:

```text
        EN     KO     JA     ZH
EN       -      .      .      .
KO       .      -      .      .
JA       .      .      -      .
ZH       .      .      .      -
```

Use retrieval metrics.

---

## Figure 3 — Procrustes improvement

```text
Raw vs Procrustes-aligned Recall@1
by layer
```

---

## Figure 4 — Language separability

```text
Language probe accuracy
vs Transformer layer
```

---

## Figure 5 — Semantic alignment vs language information

Two vertically aligned plots:

```text
retrieval
    vs layer

language classification
    vs layer
```

---

## Figure 6 — Tokenization

For each model and language, visualize one of:

```text
tokens / 100 characters
bytes / token
token count distribution
```

Keep this figure compact.

---

# 42. Figure Style

Figures should be suitable for direct inclusion in the GitHub README.

Requirements:

- descriptive titles;
- labeled axes;
- legends;
- readable font size;
- high-resolution PNG;
- optionally SVG;
- no unnecessary decorative styling;
- no 3D plots;
- no dozens of lines on the same figure.

Prefer interpretation over visual complexity.

---

# 43. Required Tables

Generate at least:

```text
results/tables/retrieval_summary.csv
results/tables/procrustes_summary.csv
results/tables/language_probe_summary.csv
results/tables/tokenization_summary.csv
```

A compact final Markdown table should summarize:

```text
Model
Best retrieval layer
Best average R@1
Final-layer R@1
Mean Procrustes gain
Final-layer language probe accuracy
```

Only populate this table from actual experimental outputs.

---

# 44. Testing

The project must contain meaningful unit tests.

Do not test Hugging Face itself.

Test our logic.

---

## `test_pooling.py`

Verify:

- padded positions are excluded from mean pooling;
- last-token pooling selects the correct token;
- output dimensions are correct.

---

## `test_retrieval.py`

Construct synthetic embeddings where translations are obvious.

Example:

```python
X = identity_matrix
Y = identity_matrix
```

Expected:

```text
Recall@1 = 1
MRR = 1
```

Then permute `Y` and verify the implementation correctly uses sentence IDs.

---

## `test_procrustes.py`

Construct:

```text
Y = X @ R
```

where `R` is a random orthogonal matrix.

Verify that learned Procrustes alignment approximately recovers perfect retrieval.

This test is extremely important because it catches matrix-orientation mistakes.

---

## `test_metrics.py`

Test manually constructed rankings.

---

## `test_data.py`

Verify:

- sentence IDs align across languages;
- requested sample count is respected;
- deterministic sampling works.

---

# 45. Code Quality

Use:

```text
Python >= 3.11
```

Core dependencies:

```text
torch
transformers
datasets
numpy
pandas
scipy
scikit-learn
matplotlib
pyyaml
tqdm
```

Development dependencies:

```text
pytest
ruff
mypy
```

Type annotate public functions.

Use docstrings where function behavior is not obvious.

Prefer small modules and pure functions for:

```text
metrics
alignment
normalization
pooling
```

---

# 46. Logging

Every experiment script should log:

```text
experiment name
model
device
dtype
number of parameters
number of layers
hidden dimension
dataset split
number of examples
languages
batch size
pooling
output location
```

At completion, report:

```text
runtime
number of processed sentences
cache path
results path
```

Avoid verbose logging inside every batch except for a progress bar.

---

# 47. Command-Line Interface

The following workflow is implemented and validated.

### Install

```bash
pip install -e ".[dev]"
```

### Smoke test

```bash
python scripts/run_all.py --config configs/smoke.yaml
```

### Main Tri experiment

```bash
python scripts/run_all.py --config configs/tri_05b.yaml
```

### Baseline

```bash
python scripts/run_all.py --config configs/qwen_05b.yaml
```

### Generate final figures

```bash
python scripts/make_final_analysis.py --config configs/tri_05b.yaml --config configs/qwen_05b.yaml
```

Use the corresponding `*_public.yaml` files in these commands when intentionally reproducing the
published mirror-based run without Hugging Face authentication.

### Tests

```bash
pytest
```

### Lint

```bash
ruff check .
```

---

# 48. `run_all.py`

`run_all.py` should orchestrate, in order:

```text
1. load configuration
2. prepare dataset indices
3. extract/cached representations if absent
4. compute raw retrieval
5. compute Procrustes retrieval
6. run language probes
7. run tokenization analysis
8. save tidy result tables
9. generate figures
```

Each stage must remain independently executable.

If a later stage fails, it should be possible to rerun it without recomputing hidden states.

---

# 49. Smoke-Test Configuration

The smoke test exists to validate the complete pipeline cheaply.

Example:

```yaml
n_train: 8
n_eval: 16

languages:
  en: eng_Latn
  ko: kor_Hang

pooling:
  - last_token
```

The goal is not meaningful metrics.

The goal is to confirm:

```text
dataset
→ model
→ hidden states
→ cache
→ retrieval
→ alignment
→ probe
→ plot
```

works end to end.

Run the smoke test before any full extraction.

---

# 50. Recommended Implementation Order

Agents working on this repository should follow this order.

Do **not** implement every component simultaneously.

## Phase 1 — Repository scaffold

Create:

```text
pyproject.toml
src/
scripts/
tests/
configs/
results/
artifacts/
```

Configure:

```text
pytest
ruff
.gitignore
```

Commit.

---

## Phase 2 — Dataset

Implement FLORES loading.

Requirements:

```text
four language columns
shared aligned IDs
deterministic sampling
dev / devtest separation
```

Add tests.

Print five multilingual examples manually during development to verify alignment.

Commit.

---

## Phase 3 — Model loading

Implement generic Hugging Face causal LM loading.

Verify Tri-0.5B first.

Run one sentence.

Inspect:

```text
number of hidden-state tensors
shape of each tensor
tokenizer special tokens
```

Do not proceed until these values make sense.

Commit.

---

## Phase 4 — Pooling and cache

Implement:

```text
last-token pooling
mean pooling
representation cache
```

Run 5–10 examples.

Verify saved shapes manually.

Expected conceptual shape:

```text
num_sentences
× num_hidden_states
× hidden_dimension
```

Commit.

---

## Phase 5 — Retrieval

Implement:

```text
cosine similarity matrix
ranking
Recall@1
Recall@5
Recall@10
MRR
```

Write synthetic tests before running a full experiment.

Run a small EN↔KO experiment.

Plot R@1 across layers.

This should produce the first scientifically interesting output of the project.

Commit.

---

## Phase 6 — Expand multilingual retrieval

Add:

```text
JA
ZH
```

Evaluate all directed language pairs.

Generate summary figures.

Commit.

---

## Phase 7 — Procrustes

Implement and test on synthetic rotated vectors first.

Then fit on FLORES dev and evaluate on devtest.

Generate:

```text
raw vs aligned
```

layer curves.

Commit.

---

## Phase 8 — Language probe

Implement logistic-regression probing.

Train on dev.

Evaluate on devtest.

Plot accuracy by layer.

Commit.

---

## Phase 9 — Tokenization analysis

Compute tokenization statistics for both models.

Generate a compact comparison.

Commit.

---

## Phase 10 — Baseline model

Only once the complete pipeline works for Tri:

```text
Qwen/Qwen2.5-0.5B
```

should be added.

The model abstraction should make this nearly configuration-only.

If adding the baseline requires rewriting the pipeline, the model abstraction is too tightly coupled.

Commit.

---

## Phase 11 — Final analysis

Generate all tables and figures.

Review results manually.

Identify only conclusions actually supported by the data.

Write the findings section.

Commit.

---

## Phase 12 — Repository polish

Ensure:

```bash
pytest
ruff check .
python scripts/run_all.py --config configs/smoke.yaml
```

all succeed from a clean environment.

Clean unused files.

Remove debug scripts.

Remove generated large files from Git history if necessary.

Finish README.

---

# 51. Three-Day Execution Plan

## Day 1 — Pipeline + first result

Target:

```text
data loading
model loading
hidden-state extraction
pooling
caching
cross-lingual retrieval
first EN↔KO layer curve
```

By the end of Day 1 there must be at least one valid scientific figure.

If there is not, reduce scope immediately.

Do not spend Day 1 on visual polish.

---

## Day 2 — Main experiments

Implement:

```text
all four languages
all retrieval directions
Procrustes
language probe
tokenization analysis
baseline extraction
```

By the end of Day 2 all core experiment tables should exist.

---

## Day 3 — Analysis + engineering quality

Focus on:

```text
tests
figures
result interpretation
README
reproducibility
code cleanup
CLI
documentation
```

Do not start major new experiments on Day 3 unless all required outputs are finished.

---

# 52. Priority Levels

## P0 — Must have

The project is not complete without:

```text
Tri-0.5B
FLORES EN/KO/JA/ZH
layer-wise extraction
last-token pooling
cross-lingual retrieval
Recall@1 / Recall@5 / MRR
reproducible caching
layer-wise retrieval figure
tests
clean README
```

---

## P1 — Strongly desired

```text
Qwen 0.5B baseline
Procrustes alignment
language probe
tokenization analysis
mean-pooling robustness check
```

---

## P2 — Nice to have

```text
bootstrap confidence intervals
representation centering
additional visualizations
interactive notebook
all pairwise languages rather than English-centric pairs
```

---

## P3 — Explicitly out of scope unless everything else is finished

```text
LoRA fine-tuning
continued pretraining
RLHF
GRPO
large models
attention-map interpretability
SAEs
activation patching
causal interventions
large benchmark suites
custom tokenizer training
distributed inference
web application
LLM chatbot UI
```

Do not sacrifice a clean P0/P1 project to implement P3 features.

---

# 53. Failure Handling

Agents must treat unexpected results as information, not implementation failures.

For example, if retrieval remains poor across all layers:

1. run synthetic retrieval tests;
2. verify sentence alignment;
3. verify pooling;
4. inspect matched vs random cosine distributions;
5. manually inspect a few nearest neighbors;
6. test mean pooling;
7. verify correct model and tokenizer;
8. only then conclude the model exhibits weak retrieval alignment under this probing method.

Never modify methodology simply because results do not look impressive.

---

# 54. Common Implementation Pitfalls

## Padding bug

Mean pooling that includes padding will corrupt representations.

Always use the attention mask.

---

## Wrong last-token index

With padded batches, `hidden[:, -1]` is not necessarily the final real token.

Use the attention mask.

---

## Procrustes leakage

Never fit Procrustes on `devtest`.

---

## Probe leakage

Do not randomly split individual multilingual vectors from the same sentence across train and test.

Use FLORES `dev` versus `devtest`.

---

## Incorrect sentence alignment

Sentence `i` must correspond to translation `i` in every language.

Preserve source IDs.

---

## Quantization by default

Do not quantize the models in the main experiment unless hardware makes it absolutely necessary.

Representation analysis can be affected by quantization.

---

## Comparing instruction and base models

Do not use Qwen Instruct as the primary baseline.

Use the base pretrained model so post-training is not an unnecessary confound.

---

## Storing full hidden states

Never cache:

```text
sentence × layer × token × dimension
```

Cache pooled sentence vectors instead.

---

## Overinterpreting cosine similarity

Higher cosine similarity does not automatically imply better semantic representation.

Primary conclusions should rely on translation retrieval and controlled comparisons.

---

# 55. Scientific Interpretation Guidelines

The final write-up must distinguish:

```text
observation
interpretation
speculation
```

Example:

### Observation

> English→Korean Recall@1 increases from layer 6 to layer 16 and decreases in the final layers.

### Interpretation

> Intermediate layers appear to expose more useful representations for translation retrieval than early or final layers under this pooling strategy.

### Speculation

> One possible explanation is that later layers become more specialized toward next-token prediction.

Only the first two should be stated confidently.

The third should explicitly be identified as a possible explanation.

---

# 56. Claims We Must Not Make

Do not claim that:

```text
one model is globally better;
one model understands Korean better;
a representation is truly language-independent;
cross-lingual retrieval proves reasoning ability;
Procrustes proves a universal semantic space;
FLORES results generalize to every domain;
tokenization efficiency alone determines model quality;
correlation across layers establishes causality.
```

The project should demonstrate good scientific judgment as much as technical skill.

---

# 57. Final Results

## Key Findings

These observations use 256 aligned FLORES `dev` sentences for fitting and 512 distinct `devtest`
sentences for evaluation, over every directed pair among EN/KO/JA/ZH. Layer 0 is the embedding
output and layers 1–24 are Transformer-block outputs. The primary representation is the final
non-padding token.

![Direction-averaged layer-wise translation retrieval](results/figures/model_retrieval_comparison_last_token.png)

1. **Translation retrieval is strongly layer-dependent.** Direction-averaged R@1 peaks at layer
   13 for Tri (0.6888) and layer 15 for Qwen (0.7705), versus a random baseline of 0.0020. It then
   falls to 0.2324 and 0.0257 respectively at the final layer. This supports an intermediate-layer
   alignment interpretation under this retrieval probe; it does not establish a general ordering
   between the models.

2. **Pooling changes the observed geometry.** Mean pooling reaches 0.7673 R@1 for Tri at layer 6,
   higher and much earlier than its last-token maximum, whereas Qwen's mean-pooling maximum is
   0.5724 at layer 15, below its last-token result. Conclusions about a single “best layer” are
   therefore representation-choice dependent.

3. **Orthogonal alignment helps unevenly across depth and model.** Averaged over all valid layers
   and directions, Procrustes changes last-token R@1 by +0.0088 for Tri and +0.0950 for Qwen. The
   largest gains occur after raw retrieval deteriorates: at layer 24 the gains are about +0.40 and
   +0.56, while at each model's best raw layer the change is small or negative. This is consistent
   with recoverable geometric structure in late representations, not proof of a universal semantic
   space.

4. **Language identity remains linearly accessible.** Final-layer probe accuracy is 0.9995 for
   both models (chance: 0.25), even though the same representations support cross-lingual retrieval
   at other depths. Semantic alignment and language separability therefore coexist in this setup.

5. **Tokenizer behavior differs materially by language.** Relative to each model's English token
   count, Tri averages 1.04×/1.20×/1.16× tokens for KO/JA/ZH, while Qwen averages
   1.65×/1.43×/1.00×. These measurements contextualize the representation results but do not imply
   that token efficiency determines model quality.

6. **The abrupt Tri pooling inversion is localized.** From layer 6 to 7, last-token R@1 rises from
   0.0695 to 0.4334 while mean-pooling R@1 falls from 0.7673 to 0.2909. Residual-stream hooks show
   that attention raises last-token retrieval to 0.4995, whereas the MLP collapses mean-pooling
   retrieval from 0.8169 to 0.2909. The mean representation's stable rank simultaneously falls
   from 14.72 to 1.01, identifying a strongly anisotropic transition.

7. **Late-layer recovery is primarily a centering effect.** At layer 24, per-language centering
   without rotation reaches 0.8184/0.8543 R@1 for Tri and 0.7212/0.6955 for Qwen under
   last-token/mean pooling. These values exceed the full Procrustes results, so the evidence points
   to large language-specific offsets rather than a benefit attributable purely to rotation.

The compact values are in [`results/tables/final_summary.csv`](results/tables/final_summary.csv),
with full retrieval, Procrustes, probe, and tokenization summaries alongside it.

### Limitations

- Results cover one deterministic sample, one seed, four languages, and FLORES sentence-level text.
  Query bootstrap intervals are reported, but no seed replication or domain-transfer evaluation is.
- Reproducing the reported run requires a Hugging Face account with access accepted for the gated
  official `facebook/flores` repository; the public fallback may not reproduce identical values.
- The analysis is observational and sensitive to pooling, model implementation, tokenizer, and
  layer convention; retrieval does not measure reasoning or establish causal explanations.

---

# 58. Suggested Final README Layout

Once the project is complete, the public-facing README should gradually become shorter than this implementation specification.

Recommended final public structure:

```text
# Project title

One-paragraph summary

Main result figure

## Why this question matters

## Method

## Models and data

## Experiment 1: layer-wise retrieval

## Experiment 2: geometric alignment

## Experiment 3: language probing

## Tokenization analysis

## Key findings

## Reproduce

## Repository structure

## Limitations

## References
```

The current document is intentionally verbose because it also serves as the implementation specification for coding agents.

---

# 59. Agent Instructions

Any coding agent working on this repository must follow these rules.

### Scientific validity takes priority over impressive-looking results.

Never fabricate, interpolate, or manually modify experimental results.

### Keep the scope small.

Do not add new experiments before required experiments work end to end.

### Prefer reusable code.

Any computation appearing in a final figure must exist in `src/`, not only in a notebook.

### Keep expensive and cheap stages separate.

Model inference should happen once.

All downstream analyses should operate on cached sentence representations.

### Validate every important mathematical component using synthetic tests.

Especially:

```text
retrieval
ranking
pooling
Procrustes
```

### Make assumptions explicit.

If an implementation detail is ambiguous, document the chosen convention.

### Do not silently recover from invalid experimental state.

For example, stale representation caches should trigger a clear warning or error.

### Do not optimize prematurely.

A correct 300-example experiment is more valuable than a complex broken 3,000-example pipeline.

### Do not rewrite working architecture without a clear benefit.

Once the experiment pipeline is stable, focus on results and documentation.

### Never commit secrets or authentication tokens.

Hugging Face tokens, if required locally, must come from environment variables or existing authenticated tooling.

---

# 60. Definition of Done

The project is considered complete when all of the following are true:

```text
[x] A fresh environment can install the repository.

[x] pytest passes.

[x] ruff check . passes.

[x] The smoke configuration executes end to end.

[x] FLORES parallel examples are loaded reproducibly.

[x] Tri-0.5B representations are extracted from every layer.

[x] Representations are cached at sentence level.

[x] Last-token and mean pooling are implemented correctly.

[x] Cross-lingual retrieval works for EN, KO, JA, and ZH.

[x] Recall@1, Recall@5, Recall@10, and MRR are produced.

[x] Layer-wise retrieval figures exist.

[x] Procrustes is fitted on dev and evaluated on devtest.

[x] Procrustes synthetic tests pass.

[x] Language probes are trained on dev and evaluated on devtest.

[x] Tokenization statistics are computed.

[x] Qwen2.5-0.5B baseline is evaluated.

[x] All final figures are generated from saved result tables.

[x] No result is manually hardcoded.

[x] The public README contains actual findings.

[x] Limitations are documented.

[x] No large unnecessary model/data artifact is committed to Git.

[x] Commands needed to reproduce the experiment are documented.

[x] Repository code is clean enough for another engineer to understand without the original author.
```

---

# 61. Success Criteria

Success does **not** mean obtaining a particular scientific conclusion.

The project is successful if it demonstrates:

### Research formulation

A clear question about multilingual representation learning.

### Experimental design

Appropriate controls, train/test separation, metrics, and comparison models.

### Transformer understanding

Correct extraction and interpretation of layer-wise hidden representations from causal LMs.

### Mathematical understanding

Cosine retrieval, ranking metrics, orthogonal Procrustes alignment, and linear probing.

### Multilingual ML awareness

Awareness of language-specific tokenization and the limitations of cross-lingual evaluation.

### Software engineering

Reusable modules, CLI scripts, testing, configuration, caching, and reproducibility.

### Resource efficiency

Useful experiments performed using sub-billion-parameter models and lightweight analysis.

### Scientific judgment

Careful distinction between empirical observations and stronger claims.

That combination is more important than producing a single high benchmark score.

---

# 62. Minimal Research Narrative

The final project should tell one coherent story:

> Multilingual language models process translations as completely different token sequences. This project investigates whether those sequences nevertheless converge toward related internal representations as they move through a decoder-only Transformer.

Then:

> We extract representations from every layer of two approximately 500M-parameter multilingual base models and evaluate whether translations can retrieve one another across English, Korean, Japanese, and Chinese.

Then:

> We test whether remaining differences between languages can be explained by a simple orthogonal transformation and whether language identity remains linearly recoverable from the same representations.

Finally:

> We complement the representation analysis with tokenizer statistics to understand how different input representations may contribute to the observed behavior.

Every experiment in the repository should contribute directly to this story.

If an experiment does not, it probably does not belong in the three-day version of the project.

---

# 63. Final Principle

This repository should feel like a **small, carefully executed research project**, not a demo.

The objective is not:

```text
“I used a multilingual language model.”
```

The objective is:

```text
“I formulated a question about multilingual language models,
designed controlled experiments to investigate it,
implemented the evaluation infrastructure cleanly,
worked within strict compute constraints,
and can explain what the results do and do not establish.”
```

That is the standard the implementation should optimize for.
