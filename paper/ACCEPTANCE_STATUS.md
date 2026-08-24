# Live acceptance status

`PAPER_ACCEPTANCE_CONTRACT.md` records the immutable acceptance assertions
defined after the planning review. This file records their current execution
state and is intentionally separate from that contract.

## Current result

- The frozen Introduction and Background hashes still satisfy A1.
- The current PDF compiles to 13 pages including references; the conclusion is
  on page 11 (the bibliography begins at the bottom of that page and continues
  through pages 12--13).
- The acceptance manifest is complete.
- The submission-level verifier is currently **blocked at A18**: the current
  citation audit is fresh but has a blocking FAIL for the frozen Nimble
  occurrence, while the claim, proof, and kill-argument artifacts are stale
  after the latest manuscript edits. The stale claim/kill/proof artifacts must
  be regenerated through their prescribed fresh-review workflows.

The stale artifacts must be regenerated through their prescribed fresh-review
workflows. Their JSON files must not be edited by hand to change hashes or
verdicts. The live command is:

```text
bash /root/.codex/aris_repo/tools/verify_paper_audits.sh paper --assurance submission
```

## Evidence portability

The checked-in `paper/data/*.json` files are compact, digest-pinned summaries.
Several provenance fields intentionally point to the raw run roots outside
this repository (for example `/workspace/latchmoe-formal-20260824` and the
three model-qualification roots). A submission artifact must package those
raw logs, manifests, output-token arrays, and release acknowledgments under an
anonymous artifact root and either preserve the recorded paths or regenerate
the summaries against that root. The manuscript does not treat an inaccessible
raw path as additional evidence.
