# Concepts

Second-brain semantic layer: nouns map atomically to file sets (EXTRACTED); verbs aggregate structural edges (INFERRED).

| Concept | Files | Mentions | Top Files |
|---------|-------|----------|-----------|
| `compute` | 2 | 3 | `app.py`, `trimario4.py` |
| `del` | 2 | 3 | `app.py`, `trimario4.py` |
| `log` | 2 | 2 | `app.py`, `trimario4.py` |

## Verb Edges

| Source | Verb | Target | Strength |
|--------|------|--------|----------|
| `compute` | `depends_on` | `del` | 1.00 |
| `compute` | `depends_on` | `log` | 1.00 |
| `del` | `depends_on` | `compute` | 1.00 |
| `del` | `depends_on` | `log` | 1.00 |
| `log` | `depends_on` | `compute` | 1.00 |
| `log` | `depends_on` | `del` | 1.00 |

## Dialectic Prompts

- Thesis: `compute` centralizes 2 files; Antithesis: `del` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `compute` centralizes 2 files; Antithesis: `log` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `del` centralizes 2 files; Antithesis: `log` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
