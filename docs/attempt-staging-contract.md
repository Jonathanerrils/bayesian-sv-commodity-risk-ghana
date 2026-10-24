# Durable SV MCMC attempt staging: implementation contract

Status: **design only**. Do not use the current checkpoint JSON as a saved posterior, and do not launch production recovery from this document.

## Existing invariant

The canonical rolling fit uses the frozen model family, strict convergence gate, fixed attempt configurations, and seeds `random_seed + attempt_no - 1`. A failed refit retains every scheduled forecast date with unavailable risk forecasts. No statistical thresholds may change.

## Job boundaries

1. **Attempt 1 job** runs exactly the first scheduled configuration for one (commodity, variant, block). It must not start attempt 2.
2. On **nonconvergence**, atomically write the completed-failed-attempt checkpoint, upload it as an Actions artifact, and exit successfully with an explicit `needs_escalation` outcome.
3. On **convergence**, produce the entire canonical block forecast in the same job, upload the block CSV and associated validation metadata, and exit with `complete`. Never attempt to reconstruct posterior draws from JSON.
4. **Attempt 2 job** runs only if `needs_escalation`. Download the failed-attempt artifact, validate data fingerprint, model version, provenance, target accept, seed, attempt schedule, and block identity, then execute only the second configuration. Produce the canonical block CSV on success or the canonical unavailable-block CSV on nonconvergence. Upload output artifacts with `if: always()`.
5. A **publish/collection job** selects exactly one block CSV, validates the expected block identity and complete forecast-date coverage, and refuses conflicting duplicates.

## Cancellation behavior

A GitHub Actions job cancelled or timed out during NUTS may not execute its upload step. Thus completed attempt 1 must be uploaded **before** starting attempt 2 in another job. An interrupted attempt 1 must be rerun; an interrupted attempt 2 must rerun only attempt 2 using the durable attempt-1 checkpoint. No mid-chain NUTS continuation is claimed.

## Provenance and outputs

Use an immutable run/campaign identifier independent of per-job GitHub run IDs; a retry must have the same campaign identifier. Include the model version and source commit SHA in the manifest, and validate the expected input-data hash and per-block seed. An uploaded checkpoint alone does not count as a completed forecast block.

## Pre-production acceptance tests

- Mocked attempt 1 nonconvergence -> artifact -> separate-runner attempt 2 -> one correct block CSV.
- Mocked attempt 1 convergence -> one correct block CSV; attempt 2 skipped.
- Cancel attempt 2 after checkpoint restore -> retry attempt 2 only.
- Reject mismatched input hash, model version, provenance, block, seed, and attempt schedule.
- Both attempts fail -> canonical unavailable forecasts with all dates retained.
- Check full CSV schema, row counts, seed/diagnostic audit, and numerical equivalence to the original algorithm for mocked deterministic fits.
- Verify no run launches the full production matrix before targeted approval.

**Do not modify the existing production recovery workflow until the staged runner and these tests exist and pass.**
