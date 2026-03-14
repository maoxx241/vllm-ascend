# Repo-local orchestrator contract

Recommended entrypoint:

```bash
bash tools/kb_inventory/run_kb_refresh.sh [--ref <git-ref>] [--with-import] [--with-runtime] [--out artifacts/kb_inventory]
```

Expected stages:

1. checkout or worktree prepare
2. static refresh
3. optional import refresh
4. optional runtime refresh
5. typed-table compile
6. dry-run fixtures
7. shadow regression
8. diff + manifest publish

Expected outputs:

- `refresh_manifest.json`
- `refresh_diff_report.md`
- `merge_readiness_gate_current.md`
- `typed_kb_tables_current.jsonl`
- `artifact_bundle.tar.gz`

Recommended return codes:

- `0`: refresh complete, no blocking regression
- `10`: refresh complete, manual review needed
- `20`: compile/fixture/shadow failure
- `30`: runtime resource unavailable
