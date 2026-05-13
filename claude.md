# Project instructions

## Core rule: keep routine tasks lightweight

For ordinary coding tasks, prefer a focused edit with minimal exploration.

Do not perform broad repository exploration, deep BoTorch/GPyTorch source tracing, or full Bayesian optimization workflow analysis unless the user explicitly asks for it or the task clearly requires it.

Examples of ordinary tasks:

- fix a syntax error
- refactor a small function
- add logging
- adjust a script path
- run a specific test
- explain a local code snippet
- make a small plotting or preprocessing change

For these tasks:

- inspect only the directly relevant project files
- run the smallest useful command or test
- report what changed and what was or was not tested

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

## Escalation rule: when to inspect BoTorch / GPyTorch internals

Only perform deep inspection of installed BoTorch and GPyTorch internals when the user is asking a serious technical, academic, or model-behavior question.

Trigger deep inspection for questions about:

- whether a prior, likelihood, kernel, transform, sampler, acquisition function, or optimizer setting is appropriate
- how `SingleTaskGP`, likelihood construction, priors, transforms, acquisition functions, or optimizer internals actually work
- why a Bayesian optimization result happened
- whether the modeling assumptions are statistically or academically defensible
- comparison between implementation behavior and theory
- debugging behavior that depends on BoTorch or GPyTorch defaults
- preparing explanations for papers, reports, theses, or serious research discussion

Do not trigger deep inspection for simple mechanical edits unless the edit directly depends on those internals.

If the task may or may not require deep inspection, choose the lighter path first. Escalate only when the correctness of the answer depends on exact BoTorch/GPyTorch implementation details or when the user asks for rigorous academic analysis.

## BoTorch and GPyTorch source inspection protocol

When deep inspection is triggered, inspect the installed BoTorch package in this exact environment instead of guessing API details from memory.

Start by confirming the installed package location:

```powershell
& 'C:\anaconda3\envs\botorch\python.exe' -c "import botorch, inspect; print(botorch.__file__)"
```

Expected location for this project environment:

```text
C:\anaconda3\envs\botorch\Lib\site-packages\botorch
```

Trace only the implementation path relevant to the user's question.

Do not stop at the top-level API when the answer depends on defaults, helper functions, priors, transforms, samplers, acquisition functions, optimizer internals, posterior behavior, or cached computations.

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

When explaining source-dependent behavior to the user, mention the specific installed source files inspected and summarize the relevant implementation path.

Prefer concrete statements such as:

```text
I inspected `...logei.py` around `qLogNoisyExpectedImprovement.__init__`, `_init_baseline`, and `_get_samples_and_objectives`.
```

Avoid generic statements such as:

```text
according to BoTorch
```

unless the relevant implementation path has actually been inspected.

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

When analyzing the Bayesian optimization workflow, connect source-code behavior back to this project's flow. For example, if `ess_bo` calls `fit_model`, and `fit_model` constructs a `SingleTaskGP`, trace how that model is built, how it is fit, how its posterior is used by the acquisition function, and how candidate generation uses the acquisition value.

Do not perform this full data-aware analysis for ordinary mechanical edits unless the edit changes model behavior, preprocessing, fitting, acquisition optimization, or candidate generation.

## Reporting expectations

For ordinary tasks, report briefly:

- files changed or inspected
- commands run
- tests run or not run
- any important limitation

For deep academic / BoTorch / GPyTorch analysis, include enough provenance for the user to verify the analysis:

- project files inspected
- installed BoTorch or GPyTorch files inspected
- important classes, functions, or parameters traced
- commands used to confirm the environment
- data or preprocessing files inspected
- limitations, such as data files not found, source files not inspected, or tests not run

Keep explanations grounded in the current environment and current project files.

If a behavior is inferred rather than directly observed in source or data, say so explicitly.

## When uncertain

If the task may or may not require deep inspection, choose the lighter path first.

Escalate only when:

- the user asks for a rigorous academic explanation
- the local code cannot be understood without installed BoTorch / GPyTorch internals
- the correctness of the answer depends on exact library defaults
- the answer will be used to justify modeling choices, experimental conclusions, or research claims

## Style preferences

Be concise for routine coding work.

Be precise and provenance-heavy for serious academic or Bayesian optimization analysis.

Do not over-explain simple edits.

Do not claim that tests, source inspection, or data inspection were performed unless they were actually performed.
