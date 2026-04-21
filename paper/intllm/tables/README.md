# IntLLM paper — Tables

Each `table_*.md` here is an authoritative markdown source for one of the
tables in the IntLLM paper (MLSys 2027 submission). Every numeric cell
in a table MUST be registered as a `Claim` in
`scripts/verify_intllm_tables.py::CLAIMS`, and the script's `--coverage`
mode scans this directory to catch drift.

Tables populated in Phase 4.2 (after Phase 2 training + Phase 3 benchmarks).

| File | Paper ref | Populated in |
|------|-----------|--------------|
| `table_2_main.md`            | §4.2 main results (PPL + zero-shot × 4 IntLLM sizes + baselines) | Phase 4.2 |
| `table_3_fp16_shadow.md`     | §4.3 fp16-shadow vs QAT gap                                      | Phase 4.2 |
| `table_4_kernel_e2e.md`      | §4.4 FajarOS Nova E2E metrics (tok/s + watts per size)           | Phase 4.2 |
| `table_5_qat_ablation.md`    | §5.1 QAT ablation (from Phase 2.4 runs)                          | Phase 4.2 |
| `table_6_arch_ablation.md`   | §5.2 arch ablation (conditional on recall trigger)               | Phase 4.2 (optional) |
| `table_7_scale_sensitivity.md` | §5.3 fixed-point scale sensitivity                              | Phase 4.2 |

## Authoring rules

1. Every floating-point literal in a markdown-table row MUST correspond to
   a `Claim` in `scripts/verify_intllm_tables.py`. CI fails if not.
2. Do NOT hand-edit numbers — write a tiny helper in the verify script
   that extracts the value from the matching `paper/intllm/results/*.json`
   and regenerate the markdown cell from the JSON.
3. Rounding: 2 decimal places for PPL, 1 decimal place for percent accuracy,
   0 decimal places for token counts. Consistency with BitNet 2B4T Table 1.
4. When adding a new table, also add a row to this README's table + a
   corresponding claim cluster in the verify script.
