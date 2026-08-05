# Claim ledger

| Claim | Evidence class | Current support | Submission gate |
|---|---|---|---|
| Fixed slots preserve addresses while expert identity changes | implementation + host contract tests | checked-in runtime and tests | retain regression coverage |
| Slot generations prevent overwrite during in-flight compute | implementation + host contract tests | checked-in lifecycle tests | add graph-path fault injection |
| Wave prefill bounds the active expert set | implementation + host contract tests | checked-in phase-split tests | add online prefill-heavy runs |
| Historical capture-off comparison | excluded diagnostic | README record only; not part of the graph-only evaluation | do not promote or rerun |
| Graph-compatible offloading improves service performance broadly | unsupported | not currently claimable | complete model/workload/capacity sensitivity matrix |

The submission matrix is graph-only. Forced eager execution and eager fallback
are rejected by the benchmark validator and are not comparison baselines.
