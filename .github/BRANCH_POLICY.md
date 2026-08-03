<!-- workspace-branch-policy: v1 -->

# Branch and merge policy

This repository follows the `llm-optimizations` workspace branch lifecycle:

- Create or reuse `feature/<project-or-optimization-name>-<purpose>` for new
  work. Do not create new `codex/...`, `feat/...`, personal-name, or generic
  `dev` branches.
- Keep one branch per coherent issue or workstream and start it from the current
  `origin/main`.
- Open a pull request to `main` when the change is reviewable. Merge promptly
  after the repository gates pass and no known blocker remains.
- Delete the feature branch after merge. Close and delete abandoned branches;
  keep a blocked branch only when it has a named owner and next gate.
- A writable runtime submodule may retain one project-specific `feature/...`
  carrier branch while the parent repository actively pins it. Delete the
  carrier after the parent moves to an upstream merge or stops using it.
