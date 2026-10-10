# Frozen study: remote execution and checkpoint transfer

Remote execution changes scheduling only. The economic model, seeds, rank,
population inputs, horizons, penalties and scientific source hashes remain those
in `rich6d_rough_sr3_extended_v2/protocol.json`. Numerical-resolution limitations
and the original scientific 5% criteria are unchanged.

Only the standard `macos-26` runner is eligible. Both Linux compatibility probes
failed exact return-history equality. A macOS history probe passed. The separate
complete fitted-policy probe must also pass before allocating production work;
its plan and tolerances were declared before the results. Its old pilot reference
predates explicit T3240 prefix keys: require every applicable saved reference
hash, including each robustness economy's terminal T3240 hash, and report added
prefix hashes without a saved reference. Exclude the baseline pilot terminal
hash because the original pilot ends at 7290 and production ends at 4860.

The five large numerical caches are downloaded from the separate input release.
Every size and SHA-256 hash must match the canonical manifest before a cache is
published. Supporting calibration, seeds and population coefficients are also
hash-checked. Remote jobs use an empty temporary output directory and never
recalibrate these inputs.

## Admission and allocation

1. Record the successful reproduction job's actual JSON result and GitHub run
   identity in `remote_reproduction_results.json`, with `job_conclusion` and
   `result` fields. Preserve failed attempts separately.
2. Stop local admission only after its current checkpoint and resource record
   have both been saved. Record the controlled handoff. Audit all local results.
3. Declare `remote_allocation_production_baseline.json`. Include the run hash,
   stage, qualification and input-manifest hashes, completed local indices, and
   consecutive batch IDs with disjoint missing indices. Completed and assigned
   indices together must equal exactly 0 through 299. Assign by index and
   operational resource limits; never select indices using outcomes.
   `python -m simulations.extended.remote_allocate --stage production_baseline`
   performs the admission checks and writes the declaration from the audited
   missing-index set. It refuses to overwrite an existing allocation.
4. Commit and push the allocation, then dispatch `rich6d-production.yml` with
   `stage=production_baseline`. The workflow uses at most five standard macOS
   runners and preserves completed checkpoints even if a batch fails.
5. Independently verify all 300 baseline checkpoints before declaring robustness
   allocation. Preserve that audit as `remote_baseline_completion.json` and bind
   its hash in `remote_allocation_production_robustness.json`. Each robustness
   index computes all four nonbaseline environments with the paired design.

The allocation and qualification checks run before remote computation. Missing
or unsuccessful qualification, changed inputs, duplicate indices, missing
indices, and premature robustness allocation are errors.

## Retrieval and independent validation

Download each workflow's small checkpoint artifacts before their one-day
retention expires. They contain full-penalty NPZ outcomes, resource records and
a transfer manifest, not the large numerical input caches. The manifest records
the exact allocation, workflow run, source commit, completed index prefix and
file checksums. A failed batch's valid completed prefix is recoverable.

Run the following with the actual downloaded directory, workflow run ID, and
source commit obtained from GitHub:

```sh
python -m simulations.extended.remote_ingest \
  --download-root /path/to/downloaded-artifacts \
  --stage production_baseline --run-id RUN_ID --commit COMMIT_SHA
```

For prompt retrieval as individual batches finish, run the collector with the
same pinned run identity and an initially empty download directory outside the
cloud-backed repository:

```sh
python -m simulations.extended.remote_collect \
  --download-root /tmp/rich6d-remote-RUN_ID \
  --stage production_baseline --run-id RUN_ID --commit COMMIT_SHA
```

The collector polls the existing run, records GitHub artifact identities,
retries transient network failures without restarting computation, and invokes
the same independent importer after new downloads. It preserves partial
results when a remote run fails and never starts robustness or claims study
completion. A failed scientific audit stops collection for inspection.

The importer acquires the original production lock to prevent concurrent local
admission. It first checks transfer provenance, checksums, grid and identities.
It then stages the candidate checkpoints together with all existing checkpoints
in a temporary directory and runs the independent scientific audit. This audit
checks analytical population ceilings and floors, normal equations, exact
regret decompositions, historical return hashes, pairing across stages and
uniqueness of economic histories. Only a successful audit permits atomic local
publication. Different existing files are never overwritten. A second audit
checks the published dataset and records the ingestion.

Partial transfers do not establish completion. Once all five environments have
300 verified replications, run the existing statistical summaries, report,
figures, numerical archive and final verification. Inspect all ten final figures
and retain the distinction between execution completion and numerical accuracy.
