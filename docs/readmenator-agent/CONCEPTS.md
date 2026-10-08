# Concepts

Nouns map atomically to file sets (EXTRACTED); verbs aggregate structural edges (INFERRED).

- `compute` | files=2 | mentions=3 | `app.py`, `trimario4.py`
- `del` | files=2 | mentions=3 | `app.py`, `trimario4.py`
- `log` | files=2 | mentions=2 | `app.py`, `trimario4.py`

## Verb Edges

- `compute` --depends_on--> `del` (strength 1.00)
- `compute` --depends_on--> `log` (strength 1.00)
- `del` --depends_on--> `compute` (strength 1.00)
- `del` --depends_on--> `log` (strength 1.00)
- `log` --depends_on--> `compute` (strength 1.00)
- `log` --depends_on--> `del` (strength 1.00)

## Dialectic

- Thesis: `compute` centralizes 2 files; Antithesis: `del` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `compute` centralizes 2 files; Antithesis: `log` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
- Thesis: `del` centralizes 2 files; Antithesis: `log` pulls 2 files with 2 shared (Jaccard 1.00); Synthesis: should they merge, split by layer, or keep `depends_on` explicit?
