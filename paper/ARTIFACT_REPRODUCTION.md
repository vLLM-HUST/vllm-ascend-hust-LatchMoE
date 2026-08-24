# Artifact reproduction notes

The paper stores the compact, digest-checked evidence summaries used to render
the tables and figures under `paper/data/`. The summaries are not a substitute
for the raw artifact bundle required for an external reproduction.

## Required raw evidence classes

1. Motivation JSONL event streams, manifests, successful-request records, and
   release acknowledgments for Qwen3-30B-A3B, GLM-4.7-Flash, and
   Qwen3-Next-80B-A3B-Instruct.
2. Qualification reports and their native/eager/graph output records.
3. Baseline, overlap, and capacity campaign unit directories, including raw
   client/server logs, output-token arrays, profile streams, HBM samples, and
   release acknowledgments.

The exact source paths and SHA-256 digests are retained in
`paper/data/formal_campaigns.json`, `paper/data/qualification_summary.json`,
`paper/data/resource_ledgers.json`, and
`paper/data/audits/motivation_reaudit.json`. In the current workspace these
roots live outside the paper directory. A portable archive of the current raw
roots was generated locally at
`paper/artifact_bundle/latchmoe_raw_20260824_v3.tar.zst`; because it is about
73 MB, the archive is intentionally excluded from Git and should be uploaded
as a release asset or object-store artifact. Its SHA-256 and file-level digest
manifest are recorded in the checked-in
`paper/artifact_bundle/artifact_manifest_v3.json`. The archive contains the
three Motivation traces, the formal baseline/overlap/capacity campaigns, the
one-request feasibility bundle, all three model qualification roots, the
Issue-7/Issue-17 evidence bundles, and the frozen workload manifest. It
contains 583 files (981,554,791 uncompressed bytes; 73,022,168 compressed
bytes). It also contains one matched eager/PIECEWISE graph qualification
diagnostic and its fixed-slot profiles; that diagnostic is not a formal
performance campaign. Extract it into an anonymous artifact directory before running the
audits; the top-level directory names preserve the source-root identity used
by the manifest.
The v3 archive SHA-256 is
`c766d81e942e718e8a29bc1111df6a3b62b6a0280426d3b34fdbad369c4baf26`.

To rebuild the manifest after replacing an artifact archive, run:

```text
python3 paper/scripts/build_artifact_manifest.py \
  --archive paper/artifact_bundle/latchmoe_raw_20260824_v3.tar.zst \
  --output paper/artifact_bundle/artifact_manifest_v3.json
```

The checked-in compact summaries remain the paper-facing source of rounded
numbers; the archive is the portable raw evidence, not a claim that the
external absolute paths in older summaries are themselves portable.

To extract the archive and rerun every deterministic audit against only the
portable files, use:

```text
python3 paper/scripts/verify_artifact_bundle.py \
  --archive paper/artifact_bundle/latchmoe_raw_20260824_v3.tar.zst \
  --workdir /tmp/latchmoe-artifact-verification
```

The wrapper writes an aggregate `artifact_verification.json` and the six
component audit outputs below the chosen work directory. In the checked
workspace this wrapper completed with status `passed` for all three Motivation
profiles, the formal campaigns, the three-model qualification matrix, the
capacity sweep, the resource ledgers, and the eager-versus-PIECEWISE
diagnostic.

## Local verification

The generated LaTeX macros and tables are derived from the checked-in compact
summaries. The manuscript-level checks are:

```text
python3 paper/scripts/build_acceptance_manifest.py .
python3 paper/scripts/check_acceptance_contract.py paper
bash /root/.codex/aris_repo/tools/verify_paper_audits.sh paper --assurance submission
```

The first command verifies artifact presence; the second enforces the page,
hash, and blocking-audit contract; the third rehashes every audit input. A
successful local build without the raw bundle does not establish external
artifact reproducibility.
