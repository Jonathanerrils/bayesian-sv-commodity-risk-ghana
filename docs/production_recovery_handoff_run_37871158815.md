# Production recovery handoff — run 37871158815

Snapshot refreshed after source artifact created at **2026-10-09 11:25:45 UTC**

Source run: `37871158815`  
Source commit: `f121fc79a203a8712ac0efb1310dccd438e0b614`  
Validated production semantics: v7, `target_accept=0.99`, strict `R-hat < 1.01`, bulk ESS `> 400`, zero divergences, ESS50 filtering, deterministic global-index seeds.

> The source run was still active at this snapshot. These classifications are exact for the artifacts available at the timestamp above. The recovery workflow always re-downloads and re-audits the source run immediately before constructing a recovery matrix, and refuses `execute_recovery=true` until the source run has completed.

## Snapshot classification

The original plan contains **662 SV refit blocks** across the three commodities and the two retained SV variants.

- **complete:** 41
- **statistical-failure:** 2
- **infrastructure-missing:** 619
- **duplicate/conflicting:** 0

Legitimate statistical failures are retained as model-availability outcomes and are **not** rerun:

- Cocoa / SV-Gaussian: blocks **10** and **13**

## Exact infrastructure-missing blocks

| Commodity | Model | Missing blocks | Count |
|---|---|---:|---:|
| Cocoa | SV-Gaussian | 7, 11, 14–98 | 87 |
| Cocoa | SV-t | 0–98 | 99 |
| Gold | SV-Gaussian | 3, 10–11, 15–116 | 105 |
| Gold | SV-t | 0–116 | 117 |
| Oil | SV-Gaussian | 11, 14–15, 22–114 | 96 |
| Oil | SV-t | 0–114 | 115 |
| **Total** |  |  | **619** |

Reusable complete blocks in the same snapshot are:

- Cocoa / SV-Gaussian: 0–6, 8–9, 12
- Gold / SV-Gaussian: 0–2, 4–9, 12–14
- Oil / SV-Gaussian: 0–10, 12–13, 16–21

No SV-t block had completed in the available artifact snapshot.

## Recovery-matrix exactness

The block audit emits recovery jobs **only** for `infrastructure-missing` blocks. Each emitted job has:

```json
{"commodity":"<commodity>","model":"<SV variant>","start_block":<block_id>,"n_blocks":1}
```

For this snapshot the matrices therefore contain:

- Cocoa: **186** unique one-block jobs
- Gold: **222** unique one-block jobs
- Oil: **211** unique one-block jobs
- Total: **619** unique one-block jobs

Thus the recovery matrix is a one-to-one cover of the 619 infrastructure-missing block keys: no complete block is recomputed, no statistical-failure block is recomputed, and no missing block appears more than once.

The audit fails closed if a block has duplicate/conflicting terminal artifacts or mismatched `target_accept` / `filter_resample_threshold` provenance.

## Recovery execution

`.github/workflows/w1000-recovery.yml` is manual and defaults to **audit-only** (`execute_recovery=false`). It must not be launched in recovery mode until the source run is completed and this infrastructure PR has been reviewed.
