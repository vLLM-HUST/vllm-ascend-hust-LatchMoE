# Claim ledger

| Claim | Evidence class | Current support | Submission gate |
|---|---|---|---|
| Fixed slots preserve addresses while expert identity changes | implementation + host contract tests | checked-in runtime and tests | retain regression coverage |
| Slot generations prevent overwrite during in-flight compute | implementation + host contract tests | checked-in lifecycle tests | add graph-path fault injection |
| Wave prefill bounds the active expert set | implementation + host contract tests | checked-in phase-split tests | add online prefill-heavy runs |
| 1.66x TTFT, 2.30x TPOT, 2.22x throughput improvement over capture-off | preliminary engineering measurement | README record only | check in raw manifests and repeat three independent starts |
| Graph-compatible offloading improves service performance broadly | unsupported | not currently claimable | complete model/workload/capacity sensitivity matrix |

The eager cases are explicit baselines and ablations. A forced eager fallback
must never be recorded as a successful graph-compatible run.
