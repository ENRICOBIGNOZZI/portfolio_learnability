# Recomputed data audit

Status: **PAYOFF_COMPLETENESS_FAILED** for the formation-only specification.

| Check | Actual result |
|---|---:|
| Verified raw files | 63 |
| Frozen features | 130 |
| Formation months | 744 |
| Formation stock-months | 2504402 |
| Missing forward payoffs | 7489 |
| Months with unresolved payoffs | 638 |
| Recovered by exact next-calendar source match | 0 |
| Finite formation-universe adjacent-calendar pairs | 2496567 |
| Lead/current discrepancies, tolerance 1e-8 | 0 |
| Rank error after all future payoffs changed to missing | 0.0 |

Counts were recomputed; none is copied into the adapter as a constant. Formation masks and ranks retain positions with missing future returns. An unknown payoff on a nonzero holding raises PAYOFF_COMPLETENESS_FAILED. The training objective requires complete monthly portfolio payoffs and also fails rather than dropping months/assets. Exact-zero holdings may exclude unknown payoffs mathematically; no learned nonzero position is zeroed.

The real sensitivity uses the existing complete-payoff sample, with exactly its filtered ranks/N_t. Every NN/affine/memory candidate receives identical data. This conditions membership and ranks on future payoff availability and is **RETROSPECTIVE_ONLY**. It cannot repair or establish a point-in-time backtest. Even a complete-payoff month in the current snapshot does not certify historical vintages.

Audit files: private_runs/audit/audit.json, counts.csv, memory.jsonl, memory.summary.json. Stock-level source files remain read-only. The original README's larger lead/current pair count covers a different (raw) universe; this audit's pair count is after formation filters and is intentionally not equated to it.
