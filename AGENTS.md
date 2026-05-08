# Project instructions

## Python environment

Always run Python code in this project with:

```powershell
C:\anaconda3\envs\botorch\python.exe
```

Do not use system Python, `python`, `py`, or another conda environment unless the user explicitly asks for it.

For scripts, use:

```powershell
& 'C:\anaconda3\envs\botorch\python.exe' script.py
```

For modules and tests, use:

```powershell
& 'C:\anaconda3\envs\botorch\python.exe' -m pytest
```

When a one-line Python command is needed, use:

```powershell
& 'C:\anaconda3\envs\botorch\python.exe' -c "..."
```

## BoTorch and GPyTorch source inspection

When editing, debugging, or analyzing code that uses BoTorch, inspect the installed BoTorch package in this exact environment instead of guessing API details from memory.

Start by confirming the installed package location:

```powershell
& 'C:\anaconda3\envs\botorch\python.exe' -c "import botorch, inspect; print(botorch.__file__)"
```

Expected location for this project environment:

```text
C:\anaconda3\envs\botorch\Lib\site-packages\botorch
```

Do not stop at the top-level API when the behavior depends on defaults, helper functions, priors, transforms, samplers, acquisition functions, or optimizer internals. Follow the relevant call chain into the installed BoTorch source code, and when necessary into the installed GPyTorch source code.

This especially applies to:

- model construction, including `SingleTaskGP`
- likelihood construction and noise models
- kernel, lengthscale, outputscale, and noise priors
- input and outcome transforms
- acquisition functions such as `qLogNoisyExpectedImprovement`
- acquisition optimizer behavior
- posterior sampling and cached root decomposition behavior
- constraints and feasibility handling

For example, when analyzing `SingleTaskGP`, inspect:

```text
C:\anaconda3\envs\botorch\Lib\site-packages\botorch\models\gp_regression.py
```

If `SingleTaskGP` calls helper functions such as `get_gaussian_likelihood_with_lognormal_prior`, also inspect:

```text
C:\anaconda3\envs\botorch\Lib\site-packages\botorch\models\utils\gpytorch_modules.py
```

If those helpers instantiate GPyTorch classes, priors, constraints, kernels, or likelihoods, inspect the corresponding installed GPyTorch source files as needed.

For acquisition-function analysis, inspect the actual installed implementation. For example, for `qLogNoisyExpectedImprovement`, inspect:

```text
C:\anaconda3\envs\botorch\Lib\site-packages\botorch\acquisition\logei.py
```

When explaining source-dependent behavior to the user, mention the specific installed source files inspected and summarize the relevant implementation path. Prefer concrete statements such as "I inspected `...logei.py` around `qLogNoisyExpectedImprovement.__init__`, `_init_baseline`, and `_get_samples_and_objectives`" over generic statements like "according to BoTorch."

## Data-aware Bayesian optimization analysis

When evaluating whether a model default, prior, likelihood, transform, acquisition setting, or optimizer setting is appropriate for this project, inspect both:

- the installed BoTorch and GPyTorch source code that defines the behavior
- the project's actual data and preprocessing pipeline

For questions about whether a prior is appropriate for a specific dataset, such as the May 5 LHS data, do not judge from the prior definition alone. Also inspect the project code that loads and preprocesses that data, including:

- input scaling or normalization
- output standardization or transformation
- sample size and dimensionality
- observed output range and variance
- apparent noise level
- outliers or failed evaluations
- whether the model is fit with fixed noise or inferred noise

When analyzing the Bayesian Optimization workflow, connect source-code behavior back to this project's flow. For example, if `ess_bo` calls `fit_model`, and `fit_model` constructs a `SingleTaskGP`, trace how that model is built, how it is fit, how its posterior is used by the acquisition function, and how candidate generation uses the acquisition value.

## Reporting expectations

When the user asks for analysis involving BoTorch behavior, include enough provenance for the user to verify the analysis:

- project files inspected
- installed BoTorch or GPyTorch files inspected
- important classes, functions, or parameters traced
- any commands used to confirm the environment
- any limitations, such as data files not found or tests not run

Keep explanations grounded in the current environment and current project files. If a behavior is inferred rather than directly observed in source or data, say so explicitly.
