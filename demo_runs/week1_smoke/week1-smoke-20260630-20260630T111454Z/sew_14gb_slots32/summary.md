# sew_14gb_slots32

- Role: main
- Status: ok_manual_benchmark
- Server log: `server.log`
- Profile JSONL: `moe_profile.jsonl`
- Benchmark JSON: `benchmark.json`

## Smoke Metrics

- Successful requests: 1
- Failed requests: 0
- Median TTFT ms: 744.159
- Median TPOT ms: 58.095
- Output throughput tok/s: 6.524

## Notes

The SEW server launched with `vllm::moe_offload_stage` included in the
PIECEWISE splitting ops, completed ACLGraph capture, reached `/v1/models`
readiness, and served one ShareGPT smoke request. The benchmark was executed
manually against the live server because the runner did not proceed past its
readiness phase after server startup.
