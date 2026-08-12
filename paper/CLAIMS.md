# Claim ledger

| Claim | Evidence class | Current support | Submission gate |
|---|---|---|---|
| Fixed slots preserve addresses while expert identity changes | runtime + fault injection + online graph evidence | 12/12 managed layers locked and replay-validated in five Issue #7 stages | retain fail-closed regression coverage |
| Slot generations prevent overwrite during in-flight compute | runtime + fault injection + online graph evidence | all recorded compute leases matched after completion; H2D publication ordering passed | extend model/concurrency matrix |
| Wave prefill bounds the active expert set without changing outputs | exact oracle + online graph evidence | three independent starts, 33/33 requests and 4,224/4,224 exact token IDs; no wave exceeded 32 slots | extend model/slot sensitivity matrix |
| Graph execution fails closed and releases its managed service | launcher/verifier fault injection + online custody evidence | explicit capture/replay, no eager fallback, release ACK and post-release HBM sample in every stage | retain fresh-checkout bundle replay |
| Historical capture-off comparison | excluded diagnostic | removed from README main results; not part of graph-only evaluation | do not promote or rerun |
| Graph-compatible offloading improves service performance broadly | unsupported | not currently claimable | complete model/workload/capacity sensitivity matrix |

The submission matrix is graph-only. Forced eager execution and eager fallback
are rejected by the benchmark validator and are not comparison baselines.
The narrow Issue #7 online result and raw artifact digest are documented in
`docs/evidence/issue-7-graph-lifecycle.md`.
