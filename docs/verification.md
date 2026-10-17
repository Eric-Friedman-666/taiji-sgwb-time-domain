# Verification of the paper release

The release reproduces the archived scientific outputs and includes the production simulation/inference code. The full 84-fit production nested-sampling campaign was not rerun. Commands in the README that explicitly say `run` or `run-pe` launch new sampling; the quickstart does not.

## Figure and table replay

`python reproduce.py all --output outputs` verifies 4,502 input/vendor SHA256 values, regenerates all 15 manuscript figures (eight main-text and seven appendix figures), writes all five numerical tables, and recomputes 84 posterior summaries. It independently calculates retained-bin full-XYZ Fisher matrices/SNRs and Bayes factors from the stored full/null evidences.

The expected result is `PASS_WITH_DOCUMENTED_DIFFERENCES`, with **245/252** original printed table numbers matching and **zero unexpected differences**. The seven differences are Table 4 SNR rounding corrections:

| log10 astrophysical amplitude | SNR column | Original integer | Direct calculation | Rounded release value |
| --- | --- | ---: | ---: | ---: |
| -11.5 | absolute | 377 | 376.411772 | 376 |
| -11.7 | absolute | 238 | 237.499772 | 237 |
| -11.9 | absolute | 151 | 149.852226 | 150 |
| -12.2 | absolute | 76 | 75.104022 | 75 |
| -12.3 | absolute | 59 | 59.657246 | 60 |
| -12.3 | relative | 52 | 52.554488 | 53 |
| -12.5 | absolute | 37 | 37.641177 | 38 |

The original notebook rescaled previously rounded legacy SNRs by `1/sqrt(2)`. The release instead evaluates each SNR before rounding. All printed Bayes-factor entries agree. The optional [Table 4 patch](../reference/table4_rounding_correction.patch) records the corresponding manuscript edits; running the code does not edit a manuscript.

Replay passed on Linux and in a fresh macOS Python 3.12.7 environment, including a run with original-project file reads and network access blocked. Linux replay was also checked with LaTeX typography. Figure pixels can depend on fonts and Matplotlib/TeX versions; the numerical checks, rather than image byte identity, define the acceptance criteria. Machine-readable summaries are in `reference/tested_replay_*.json`.

## Simulation and likelihood checks

The retained observations for all 84 fits were checked at the original truth parameters, without sampling. For 82 ordinary cases the absolute log-likelihood tolerance is `1e-5`; the largest observed error was `6.974e-6`. The two intentional unequal-arm-data/equal-arm-model mismatch fits use `1e-3`: errors were approximately `3.09e-4` and `2.44e-4` on totals of order `3e8`. Their covariance condition numbers reach approximately `2.47e7`, and the unmodified original likelihood itself differs between CPU/GPU builds by order `1e-4`. The exported input arrays were compared directly with the original arrays. See [mismatch_precision.json](../reference/mismatch_precision.json) and [tested_all_inference_inputs.json](../reference/tested_all_inference_inputs.json).

| Check | Scope | Relative Frobenius error |
| --- | --- | ---: |
| FD astrophysical simulation | All 360 segments generated; array comparison at segments 0, 1 and 359, 3,419 retained bins and all nine covariance entries; full-data likelihood checked | 2.103e-17 |
| Equal-arm TD astrophysical simulation | Complete first one-day simulation; all 3,419 retained bins and nine covariance entries | 4.763e-12 |
| Unequal-arm TD sound-wave simulation | Complete first one-day simulation; all 3,419 retained bins and nine covariance entries | 1.933e-12 |
| Four-parameter TD realization 1 | Complete first one-day simulation, seed 20360506; all 3,281 retained bins and nine covariance entries | 1.725e-12 |

All 18 analytic Fisher derivatives (eight- and ten-parameter models) were compared with finite differences; the largest relative Frobenius error was `2.096e-8`. These are bounded numerical checks, not exhaustive regeneration of every year-long TD observation. Reports are in [tested_pipeline.json](../reference/tested_pipeline.json) and [tested_four_parameter_simulation.json](../reference/tested_four_parameter_simulation.json).

Small four-/eight-parameter sampler execution checks were also completed during preparation. Their deliberately sparse samples do not establish posterior convergence or evidence accuracy. No new production PE result is claimed. New inference is stochastic; compare posterior/evidence estimates using their uncertainties and assess convergence for the intended run.

The optional `validate_pipeline.py` and `infer.py check` commands can repeat numerical checks when the required full input archive or regenerated bundle is available. Only the compact replay data are hosted with this release.
