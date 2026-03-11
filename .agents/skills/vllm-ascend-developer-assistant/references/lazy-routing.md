# Lazy Routing Rules

Use this file to keep the first hop cheap.

## First-Hop Principle

On the first reply, classify the task using only:

- user wording
- explicit file paths
- explicit metrics or error signatures
- `../_shared/INDEX.md`
- `../_shared/task-index.md`

Do not load heavyweight generated indexes during first-hop routing unless the user already supplied concrete evidence that requires them.

## Heavy Files

Treat these as heavyweight and load them only on demand:

- `../_shared/knowledge-governance/generated/task_skill_index.json`
- `../_shared/knowledge-governance/generated/design_analysis_index.json`
- `../_shared/knowledge-governance/generated/imported_knowledge_manifest.json`

`imported_knowledge_report.json` is allowed when you need a small confidence or coverage summary.

## Routing Gates

### Performance analysis

- If the user only says they want profiling or perf analysis, do not load heavy files.
- Return:
  - primary task type
  - route chain
  - minimum evidence checklist
  - next prompt template
- Load heavier perf knowledge only after the user provides metrics, traces, or file paths.

### Debugging

- If there is no log line, stack trace, or failure phase, do not load heavy files.
- Ask for the smallest evidence bundle first.

### Design analysis or model adaptation

- Load `design_analysis_index.json` only after the user names a concrete design surface, API, model family, or code area.

### Code-path or symbol questions

- Load `imported_knowledge_manifest.json` only when the request asks for code locations, APIs, symbols, or repo ownership.

## Low-Context Output Pattern

When evidence is thin, stop after:

- `Primary task type`
- `Route chain`
- `What is missing`
- `Smallest next step`
