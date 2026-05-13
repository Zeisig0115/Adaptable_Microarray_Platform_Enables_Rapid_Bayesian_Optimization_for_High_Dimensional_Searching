# Project instructions

This is a Bayesian optimization project built on BoTorch / GPyTorch, used to design enzyme/substrate
recipes (TMB + H2O2 + a set of additives). Two parallel BO workflows coexist: a 2D essentials-only
loop with replicate-aware diagnostics, and a 26D flat-encoded loop over the full additive set.

## Repository map

Root-level Python entrypoints — each is **independent**, not a library:

| File | Role | When to touch |
| --- | --- | --- |
| `ess_bo.py` | TMB + H2O2 BO, **keeps replicates** (one condition can have many measurements). Group-LOO + kernel-condition diagnostics. Default surrogate: `SingleTaskGP` with `matern_with_hvarfner_prior` from `fit_model.py`. | Anything about the 2-D essentials loop, replicate handling, or the May-5 LHS / HRP-titration data. |
| `add_bo.py` | 26-D flat-encoded BO over essentials + ~24 additives. **Canonical comparator** between the old `MixedSingleTaskGP` (Hamming kernel) and the custom `AdditiveSetKernel` (Tanimoto active-set + per-additive RBF). Runs through profiles in `bo_profiles/*.json` for diagnostic vs early-stage vs fast vs strong runs. | Anything about additive selection, candidate generation across the full additive set, kernel comparison, or BO profiles. |
| `add_bo_mod.py` | Variant of `add_bo.py` with a **chemistry-prior mean + kernel** (family descriptors: PEG / polymer / surfactant / polyol-sugar / protein / salt / chloride / sulfate / chelator / solvent + log MW). Goal: stronger inductive bias on higher-cardinality recipes. **Currently in validation, not the canonical pipeline** — do not assume results here override `add_bo.py`. | Only when the user explicitly mentions `add_bo_mod`, family-aware priors, or hierarchical additive structure. |
| `fit_model.py` | Library for `ess_bo.py`. Provides `matern_with_hvarfner_prior`, `fit_gp` (handles `SingleTaskGP` and `MixedSingleTaskGP`), plus NUTS variants `fit_saasbo` / `fit_fullyb_gp` / `fit_saas_gp`. | Anything that changes prior, likelihood, kernel construction, or fitting routine for the essentials loop. |
| `LHS.py` | One-shot Latin Hypercube sampler in log-space (`scipy.stats.qmc`). Generates the initial design CSV. | Only when changing how a new initial sample set is drawn. |
| `fixed_noise_gp_probe.py` | One-off noise-sensitivity probe using a fixed-noise GP on the essentials data. | Only when explicitly debugging noise assumptions. |
| `data_loader_merge.py`, `data_loader_objectives.py` | Preprocess raw frame-level measurements into BO inputs (Hampel spike removal, AUC / intensity extraction). | Only when changing data ingestion. |
| `archive/flat_encoding.py` | Historical version of the flat-encoding pipeline. **Do not edit** unless asked. | Reference only. |

`bo_profiles/` holds named argparse presets for `add_bo.py` (`default`, `early_stage_candidates`,
`fast_candidates`, `strong_candidates`). Switch with `--bo_profile <name>`.

## Data and logs

- **No data file lives at the repo root.** All experimental data and run outputs live under `logs/`.
- Filenames vary across batches and are not stable. **Treat `data_corrected.xlsx` as a schema, not a
  fixed name** — the current convention is "one row per recipe, target column directly readable by
  `add_bo.py` / `add_bo_mod.py`." When in doubt about which file to use, ask. Until further notice,
  assume the working data is in `data_corrected` form inside the most recent `May_XX_full_log/`.
- `logs/May_05_full_log/05_05_LHS_HRP_*.xlsx` are the May-5 LHS measurements at different HRP
  concentrations — these are what `ess_bo.py` reads by default.
- **Forward-looking pipeline (not finalized):** lab delivers raw plate data like `05_05_LHS.xlsx`,
  `data_loader_merge.py` aggregates replicates into per-condition mean + variance, and a
  `FixedNoiseGP` consumes them. The design is still open — do not assume this path is in use
  unless the user explicitly says so.
- Per-day folders follow `May_DD_full_log/` (also `Apr_29_full_log/`). Place all run outputs
  (candidates CSVs, diagnostic JSONs, plots, run logs) for a given day inside the matching folder.
- For a focused sub-experiment on a given day, create a subfolder under that day, e.g.
  `logs/May_12_full_log/kernel_similarity_comparison/`. Do **not** create new top-level folders
  under `logs/` for one-off experiments.

## Python environment

Always run Python in this project with:

```powershell
& 'C:\anaconda3\envs\botorch\python.exe' script.py
& 'C:\anaconda3\envs\botorch\python.exe' -m pytest
& 'C:\anaconda3\envs\botorch\python.exe' -c "..."
```

Do not fall back to `python`, `py`, or another conda env.

## How to choose your working depth

This project mixes mechanical edits with rigorous Bayesian-modeling analysis. Pick the lighter
mode unless one of the deep-mode triggers below fires.

**Light mode** (most tasks): inspect only the directly relevant project file(s), make the change,
run the smallest useful command or test, report what changed.

Use light mode for: syntax fixes, small refactors, logging tweaks, path adjustments, running an
existing test, explaining a local snippet, plotting tweaks, preprocessing edits, profile JSON
edits.

**Deep mode**: also read the installed BoTorch / GPyTorch source, plus the actual data and
preprocessing the question depends on.

Trigger deep mode whenever any of these is true:

- The user asks about a specific BoTorch / GPyTorch function, class, argument, or default
  (e.g. "what does `prune_baseline=True` do in `qLogNEI`?", "is the default likelihood prior
  appropriate?", "how does `SingleTaskGP` build its kernel when I pass `covar_module=None`?").
  **Default to reading the installed source — do not answer from memory.**
- The user asks whether a prior, likelihood, kernel, transform, sampler, acquisition setting,
  or optimizer setting is appropriate for this dataset.
- The user asks *why* a BO result happened, or whether modeling assumptions are statistically
  defensible.
- The answer will be used in a paper, report, thesis, or to justify an experimental decision.
- The local code cannot be understood without reading the installed library.

If a task could go either way, start light; escalate only when correctness depends on exact
library internals.

## BoTorch / GPyTorch source inspection protocol

When deep mode fires, read the installed source in this env — do not paraphrase from memory.

```powershell
& 'C:\anaconda3\envs\botorch\python.exe' -c "import botorch, inspect; print(botorch.__file__)"
```

Expected root: `C:\anaconda3\envs\botorch\Lib\site-packages\botorch`.

Trace the full implementation path the question depends on. Do not stop at the public API when
the answer hinges on defaults, helper functions, priors, transforms, samplers, posterior
behavior, or cached computations. Common entry points for this project:

- `botorch\models\gp_regression.py` — `SingleTaskGP`, `MixedSingleTaskGP`
- `botorch\models\utils\gpytorch_modules.py` — likelihood / prior helpers invoked by the above
- `botorch\acquisition\logei.py` — `qLogNoisyExpectedImprovement` (including `_init_baseline`,
  `_get_samples_and_objectives`), `qLogExpectedImprovement`
- `botorch\optim\optimize.py` — `optimize_acqf`, `optimize_acqf_mixed`
- `botorch\models\fully_bayesian.py` and `botorch\models\map_saas.py` — SAASBO / fully-Bayesian paths used by `fit_model.py`
- Follow into `gpytorch\...` for prior / kernel / constraint internals as needed.

When explaining source-dependent behavior, cite the specific files and functions inspected.
Prefer "I inspected `logei.py` around `qLogNoisyExpectedImprovement.__init__`, `_init_baseline`,
and `_get_samples_and_objectives`" over a generic "according to BoTorch".

## Data-aware analysis

For questions about whether a prior / likelihood / transform / acquisition setting is appropriate
**for this dataset**, do not judge from the library definition alone. Also inspect:

- the project loading and preprocessing code (`data_loader_*.py`, the `_prepare_data` /
  `load_training_data` helpers inside `ess_bo.py` / `add_bo.py`)
- input scaling: `ess_bo.py` does `log10` on essentials before normalizing; `add_bo.py` uses a
  flat `(binary, log10 concentration)` codec
- outcome transform: both pipelines use `Standardize(m=1)`
- the actual `data_corrected.xlsx` / `05_05_LHS_HRP_*.xlsx` content the run is reading: sample
  size, dimensionality, observed range, variance, apparent noise, outliers, fixed vs inferred
  noise

Trace from the BO workflow down to the source: e.g. `ess_bo.EssentialsBO.__init__` →
`fit_model.fit_gp` → `SingleTaskGP(covar_module=matern_with_hvarfner_prior(...))` → installed
GPyTorch.

## Reporting

For light-mode work, report briefly: files changed, commands run, tests run or skipped, any
limitation.

For deep-mode work, include enough provenance to verify the analysis:

- project files inspected
- installed BoTorch / GPyTorch files inspected (with the specific functions traced)
- data / preprocessing files inspected
- commands used to confirm the environment
- anything inferred rather than directly observed

Do not claim that source inspection, data inspection, or tests were performed unless they
actually were. If something is inferred, say so.

## Scope of changes

Touch only what the task requires. Do not refactor adjacent code, restyle existing comments
or `print` strings, or delete code you did not write — even in `archive/`, in `add_bo_mod.py`
(still in validation), or in scripts that look unfinished. If you spot an unrelated issue,
mention it instead of fixing it.

## Style

- Be concise for routine work. Do not over-explain simple edits.
- Be precise and provenance-heavy for serious BoTorch / GPyTorch analysis.
- **Code and plots are English-only.** Identifiers, comments, docstrings, `print` strings, log
  messages, plot titles / axis labels / legends / annotations — all in English. This applies to
  new code; do not retranslate existing Chinese strings unless the user asks. Chat replies to the
  user follow the user's language.
