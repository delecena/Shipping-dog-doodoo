# Notebook validation

Validated on September 16, 2026 using the project's Python 3.12.4 environment.

## Results

| Notebook | Completed solution | Unfinished exercise |
|---|---|---|
| 01 · Models, messages, and schemas | Passed in a fresh kernel | Expected Challenge 1 `NotImplementedError` |
| 02 · Retrieval and tools | Passed in a fresh kernel | Expected Challenge 1 `NotImplementedError` |
| 03 · LangGraph workflows | Passed in a fresh kernel | Expected Challenge 1 `NotImplementedError` |
| 04 · Recovery and review | Passed in a fresh kernel | Expected Challenge 1 `NotImplementedError` |
| 05 · Knowledge graphs | Passed in a fresh kernel | Expected Challenge 1 `NotImplementedError` |
| 06 · Research-assistant capstone | Passed in a fresh kernel | Expected Challenge 1 `NotImplementedError` |

All 12 files passed notebook-format validation and Python syntax parsing. Each was executed in its own kernel with a temporary working directory, demonstrating that no other notebook, copied script, or local fixture file was required. Exercise execution stopped at its first intentional acceptance failure; completed solutions ran through every cell.

The checks exercised:

- Typed message structure and strict schema rejection, independently from source support.
- Deterministic retrieval, ranking ties, invalid limits, missing matches, copied results, and LangChain tool invocation.
- Explicit graph nodes, append reducers, empty-evidence routing, rejected citations, successful repair, and a two-draft stopping limit.
- Temporary failure recovery, exhaustion after three attempts, and immediate propagation of non-retryable errors.
- Review payloads, approval, rejection, and resumption through a newly compiled graph sharing the original saver and thread ID, without repeating retrieval.
- Typed two-hop traversal, multiple maintainers, duplicate paths, stable ordering, missing provenance, and fixture preservation.
- Capstone coverage, relevance, completeness, unsupported budgets despite nonempty retrieval, false statements with real citations, incomplete multi-source answers, and explicit operational-error results.

The optional model sections remained disabled. API configuration was blanked in validation kernels, and tracing was disabled. No paid provider calls were made. Live-provider responses and account-specific model compatibility remain untested.

Only in-process checkpoint continuation was tested. Persistence after restarting Python, real internet retrieval, graph databases, and automatic knowledge extraction are outside these exercises.

## Delivery checks

- All delivered code-cell outputs are empty and execution counts are unset.
- Exercise copies contain no solution-tagged cells.
- Human review uses separate pause/resume cells, with no blocking `input()` calls.
- README links to the notebooks, solutions, and guide resolve locally.
- `uv lock --check --offline` passed for the updated notebook dependency group.
- Existing application code, credentials, and the learning guide were not edited.

Notebook validation used nbformat and nbclient. Resolved runtime versions are recorded in the project's `uv.lock`.
