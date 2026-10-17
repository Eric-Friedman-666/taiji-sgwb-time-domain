# Simulation and inference details

Start with the figure/table quickstart in [README.md](../README.md). The code and compact release data suffice for that workflow; no private server account, rendered figure input, or sampling run is required.

## Fit definitions and settings

`python infer.py list` lists all 84 fits. [The fit index](fit-index.md) maps each ID to its original posterior filename. [reference/inference_manifest.json](../reference/inference_manifest.json) gives the fitted parameter names, production sampler settings, source attributes, and array shapes. [data/cases.json](../data/cases.json) identifies the selected benchmark/scan cases and their simulation seeds and injected parameters. Simulation seeds and sampler seeds have different roles: the former identify the observation; the wrapper's `--seed` controls new nested sampling.

| Fitted dimensions | Fits | nlive | dlogz | walks |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 10 | 1000 | 0.5 | 32 |
| 6 | 34 | 1000 | 0.1 | 64 |
| 8 | 37 | 1000 | 0.1 | 64 |
| 10 | 3 | 1000 | 0.1 | 64 |

The six-parameter null model removes the astrophysical component. The sound-wave null retains the eight-parameter noise/DWD/astrophysical model. Truth projection, priors and likelihood normalization are implemented in `infer.py`; the original drivers remain available for comparison. The wrapper uses float64 full complex-XYZ covariances and preserves the left-conjugated data convention. It uses the exact sum of observations for segment-independent FD tables to avoid repeating identical matrix solves; this changes floating-point summation order only.

## Regenerated input bundles

For a regenerated self-contained bundle, use:

```bash
python infer.py check --case fit_011 --bundle /path/to/bundle.h5
python infer.py run --case fit_011 --bundle /path/to/bundle.h5 --seed 20261017
```

The case must identify the corresponding original science configuration. The wrapper checks the input shape and original truth likelihood before fitting. A bundle must contain `frequency`, `parameters_true`, `data/obs_csd_star`, and the response/acceleration/OMS arrays under `tables/`, together with the original normalization attributes. Fresh results go to `inference_outputs/fit_NNN/posterior.h5` by default. They link to the input bundle, so keep that bundle with the results. Archived figure/table replay continues to use the archived case selection; it does not silently substitute fresh samples.

The eight-/ten-parameter driver build subcommands accept the realization, simulation seed, injected amplitude, observation duration, frequency cuts, notch width, response seed and sky sample count. Use the values associated with the desired original posterior. The astrophysical and sound-wave TD benchmarks use the provided daily equal-/unequal-arm orbit inputs and the detector cache described in the README. The FD build commands use the provided FFT grid and static detector table. A build can create a new table when no matching table is cached.

## Four-parameter realization series

The original series has ten realizations, each containing ten one-day segments. Segment `s` (zero-based) of realization `r` uses `20260506 + 100000*r + s`. To regenerate realization 1, its TD detector table and its full-XYZ bundle:

```bash
MODE=generate_one GEN_REALIZATION_INDEX=1 python scripts/astro/xyz4_multirealization_fullcov.py
python scripts/astro/ast4_control_diagnostics_20260508.py build-td-table --treatment tenseg_1day --source-realization 1
python scripts/astro/ast4_control_diagnostics_20260508.py build-td-mask --treatment tenseg_1day --source-realization 1
python scripts/astro/ast4_control_diagnostics_20260508.py bundle-td-full4 --treatment tenseg_1day --method td_full_xyz --realization 1
```

Read the printed bundle path and pass it to `infer.py check --case fit_001 --bundle ...`. Repeat with realization/case indices 0 through 9. These commands rebuild observations and tables; they do not launch PE. The source attributes in the manifest preserve the final production masks/settings, while generated filenames can differ from the archived campaign names.

## Optional full inference archive

The approximately 5.7 GB full inference archive is **not hosted in this release**. Its checksums are retained in [data/releases.json](../data/releases.json) for provenance. If a separate copy is available, the alternate workflow is:

```bash
python get_data.py --kind inference --archive /path/to/lisa_td_paper_inference_data_20261017.tar.gz
python infer.py verify-data
python infer.py check
python infer.py run --case fit_011 --output inference_outputs
```

The archive contains `data/inference_inputs.h5`, with each fit under `fits/fit_NNN`. Repeated observations/tables are deduplicated; the hash and full file size are in the inference manifest. `infer.py check` evaluates likelihoods only. `infer.py run --all` would execute all 84 production samplers sequentially and is intentionally excluded from the quickstart.

## Included source scope

The [source list](SOURCE_LIST.txt) identifies the scientific modules and nineteen necessary plotting cells. Component-validation generators are included because their outputs appear in appendix figures. Exploratory notebooks, unused validation campaigns, scheduler logs, temporary test outputs, private manuscript sources, and maintainer-only export tools are omitted.

The `reference/` directory records source hashes, original runtime/settings, figure mappings, expected table values, portability patches, and numerical verification reports. It is provenance/acceptance data rather than a substitute for the scientific calculations. The full list of tested numerical scopes and known rounding differences is in [verification.md](verification.md).
