FidRef rebuttal supplement — 2026-11-14, English documentation v2

1. Relationship to the submitted release

The original anonymous repository revision
16815afa9f9533ddf10fe01b6000d1120cb83b1c and submission ZIP remain preserved.
Extract this supplement into a separate directory. This English documentation
edition preserves the scientific files of rebuttal v1 byte for byte; it updates
the README and release metadata.

2. Contents

- 540 result files omitted from the submission package: 480 budget-sweep runs
  and 60 historical Identifiable runs.
- Two cited files: analysis/sweep_figure_20260910/all_cells.csv and
  paper/iclr2027/supplement/metrics_detail.pdf.
- 661 original and follow-up result records, together with initial candidates,
  call records, and resume checkpoints:
  Deployment information conditions: 360; Retail information conditions: 120;
  Deployment truncation follow-up: 36; Retail truncation follow-up: 15 and
  unsent-request follow-up: 7; Identifiable original runs: 40, total-budget
  follow-up: 40, and final-output follow-up: 3; basic Shopping B24:
  20 each for 14B and 32B.
- Non-LLM diagnostic: symbolic synthesis at 139 request boundaries conditional
  on 40 original observation traces, with 80 LLM/symbolic contract evaluations.
  source_spend/source_usage record the cost of collecting the observations.
- Domain implementations, DSLs, prompts, experiment kernels, frozen Retail
  source, and the non-LLM learner.

661 counts stored original and follow-up result files. Follow-ups map back to
their source trials; the base denominator remains 20 per condition. Development
and smoke runs are separate.

3. Verification

Using Python 3.10 or later, run from the top level of the extracted directory:

    python3 -B verify_rebuttal.py

The verifier runs offline with zero model calls. It checks file SHA-256 hashes,
the missing-file list, Exact/IoU/CFR for 1,201 results, 24 sweep cells, Deployment
information conditions, Retail/Identifiable replacement policies, B24, and
deterministic symbolic replay at all 139 request boundaries.

IoU = 100(A-FR)/(A+FA), CFR = 100(A-FR-PV)/(A+FA).
A is the number of states with successful execution. Cell denominators retain
all 20 trials, including parsing failures. A trial without an evaluable contract
contributes zero to the all-trial mean.

4. Result and policy locations

All paths below are relative to analysis/rebuttal_20261106/.

- deployment_results_20261113/summary.json: 18 Deployment cells and the
  postcondition fault audit.
- info_launch_20261113_v2/main/validation/: condition-specific full-closure
  evaluations and effect audits.
- reply_draft_20261113/LATEST_INTEGRATION.json: 120 original Retail runs and
  mappings for 22 replacements.
- identifiable_length3_20261114/integrated_summary.json: per-seed paths and
  metrics for original, total-budget, and final-output follow-ups.
- nonllm_main_20261114/run/: observation packets, learner outputs, evaluations,
  and aggregates.
- reply_review_20261114/RETRY_CELLS.csv: truncation-replaced main-table and
  ablation cells with before/after metrics.
- reply_review_20261114/MISSING_FILES.csv: original hashes of the 542 files
  omitted from the submission package.

Cohort manifest.json files and raw results preserve protocols and budgets.
Call records and replacement mappings identify the earlier responses reused
by each continuation.

5. Corrections

The two historical Table 9 rows use full-closure oracle-assisted
counterexample_guided synthesis. The new B24 cohort uses sampled Active;
all 40 runs have oracle_feedback_queries=0.
The historical Identifiable method is No-Coverage, with median FA 20 at the
base budget and 12 at twice the budget. New Full Active and No-Coverage runs
form separate cohorts. Result metadata records each model, serving configuration,
decoding setting, and budget.

6. Anonymization and hashes

In RELEASE_MANIFEST.json, source_sha256 identifies original bytes and sha256
identifies distributed bytes. Operational metadata such as local paths, user
identifiers, and endpoints has been anonymized. Request messages, responses,
contracts, evaluations, budgets, and usage were checked against the originals.
Historical source/protocol hashes inside experimental records retain their
original values. Use the manifest's source_sha256 mapping when relating those
values to the distributed files.

Server startup, resource cleanup, notification, and authentication tools are
internal operational utilities. This supplement's execution entry point is
the offline verifier of stored results. Experiment kernels and frozen source
support review of the algorithms and information policies.
