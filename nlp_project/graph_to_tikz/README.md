# graph_to_tikz

Converts DAG-style QA decomposition graphs to standalone LaTeX/TikZ. Validates that the graph is a DAG. Same node style for Retrieval and Reasoning; node text is wrapped, not truncated.

**Graph format** — list of dicts, each with: `NodeID`, `Sub-Level-Question`, `Action`, `Top k`, `Next` (list of node IDs).

## API

```python
from graph_to_tikz import graph_to_tikz, load_graph, parse_fewshot_file

tex = graph_to_tikz(graph, title="Example 1")
# graph: list of dicts, JSON path, or JSON string
```

- `load_graph(graph)` → list of `DagNode` (validates DAG).
- `parse_fewshot_file(path)` → list of `(title, graph)` from fewshot `.txt` (Example N / Output blocks).

## CLI

**Single graph (JSON):**

```bash
python graph_to_tikz.py input.json output.tex [--title "Example 1"]
```

**Fewshot file (one .tex per example):**

```bash
python graph_to_tikz.py --fewshot fewshot_parallel.txt out_dir
```

Options: `--level-gap-cm`, `--node-gap-cm`, `--no-standalone`, `--no-compile` (by default pdflatex is run on each generated .tex).
