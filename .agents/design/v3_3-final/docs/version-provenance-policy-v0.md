# Version provenance policy v0

## Decision

During the current KB construction phase, exact branch/commit equality between:

- the static analysis workspace, and
- the live runtime environment used for probe collection

is **not** required.

Both may be treated as `current_mainline` evidence sources, provided that provenance is preserved.

## Required provenance fields

Every evidence item should be able to carry:

- `evidence_source`
- `repo_root`
- `module_path`
- `git_detected`
- `git_state`
- `captured_at`
- `runtime_tuple_id`
- `compat_scope`

## Source precedence for current phase

When sources disagree, use this precedence order:

1. `runtime_probe`
2. `runtime_import`
3. `repo_static`
4. `operator_note`

unless a lower-priority source is the only one available, in which case it must remain explicitly labeled.

## Not in scope yet

This policy does **not** solve:

- historical version compatibility,
- branch-aware rule selection,
- version-conditioned support matrices.

Those belong to a later multi-version KB phase.
