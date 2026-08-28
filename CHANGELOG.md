# Changelog

All notable user-facing changes to AttackBenchLib are documented here.

## 2.0.1 (2026-08-28)

### Added

- Restored the `stutz_2020` and `xiao_2020` model registry entries using new,
  independently implemented MIT architecture definitions. No upstream source or
  checkpoint is bundled.
- Stutz's author-hosted checkpoint is available only after explicit acknowledgement of
  its noncommercial terms. Xiao's checkpoint must be supplied by the user because its
  upstream repository states no license. Both loaders verify SHA-256 checksums and use
  restricted weights-only deserialization.

## 2.0.0

AttackBenchLib 2.0.0 corrects the benchmark protocol and is intentionally not
result-compatible with 1.x.

### Changed

- `run_attack()` now returns the smallest successful perturbation observed during
  optimization (`d*`), while the returned iterate is retained separately in
  `final_distances` for diagnostics.
- A 2,000 forward-plus-backward propagation budget is enforced by default. Pass
  `query_budget=None` only for exploratory runs that need not be paper-comparable.
- Query counts, timings, predictions, hashes, and protocol failure indicators are
  always included in results; the old `include_metadata` switch was removed.
- Cached W&B artifacts are accepted only when their complete 2.0 schema, sample
  hashes, and query budget match the requested run.
- Attack discovery is generated from the installed implementations and reports only
  attack/norm combinations that can actually be constructed.
- Fixed-budget preconfigured attacks now choose a norm-appropriate default epsilon,
  and FMN uses norm-specific step-size defaults.
- The package supports Python 3.9 through 3.13 and keeps attack, model, metrics,
  documentation, and development dependencies in separate extras.
- DeepFool, sparse PGD-L0, and the FMN L1 projection are now independent
  implementations based on their published algorithms rather than unlicensed source
  adaptations.
- The legacy `stutz_2020` and `xiao_2020` bundled model entries were removed because
  their source terms are not compatible with an MIT-only distribution. The Wang 2023
  registry keys now use the corresponding RobustBench models.

### Added

- `attackbench-acceptance` (and `python -m attackbench.acceptance`) for bounded,
  reproducible release and paper-protocol checks.
- Hash-matched local/global optimality helpers and complete precompiled-result
  storage.
- CI coverage for the Python version floor/current versions, optional attack and
  model integrations, warning-free documentation, and built distributions.
- A trusted-publishing workflow for TestPyPI and PyPI. Publishing remains an explicit
  maintainer action documented in `docs/releasing.rst`.

### Migration notes

- Recompute and re-upload lower-envelope artifacts before comparing 2.0 results with
  the published leaderboard: pre-2.0 artifacts contain last-iterate distances.
- Install `attackbenchlib[attacks]` for FMN and third-party attack wrappers, and
  `attackbenchlib[models]` for RobustBench checkpoints.
- Treat 1.x and 2.0 optimality/security-curve values as different protocols; do not
  combine them in the same result table.
