from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union


JsonLike = Union[str, Path, Sequence[Mapping[str, object]]]


@dataclass(frozen=True)
class DagNode:
    node_id: int
    question: str
    action: str
    top_k: str
    next_ids: Tuple[int, ...] = field(default_factory=tuple)

    @staticmethod
    def from_mapping(obj: Mapping[str, object]) -> "DagNode":
        try:
            node_id = int(obj["NodeID"])
            question = str(obj["Sub-Level-Question"])
            action = str(obj["Action"])
            top_k = str(obj["Top k"])
        except KeyError as exc:
            raise ValueError(f"Missing required key: {exc}") from exc

        next_raw = obj.get("Next", [])
        if not isinstance(next_raw, (list, tuple)):
            raise ValueError(f"Node {node_id}: 'Next' must be a list or tuple.")

        try:
            next_ids = tuple(int(x) for x in next_raw)
        except Exception as exc:
            raise ValueError(f"Node {node_id}: invalid entry in 'Next': {next_raw}") from exc

        return DagNode(
            node_id=node_id,
            question=question,
            action=action,
            top_k=top_k,
            next_ids=next_ids,
        )


def load_graph(graph: JsonLike) -> List[DagNode]:
    if isinstance(graph, Path):
        raw = json.loads(graph.read_text(encoding="utf-8"))
    elif isinstance(graph, str):
        text = graph.strip()
        if text.startswith("["):
            raw = json.loads(text)
        else:
            raw = json.loads(Path(text).read_text(encoding="utf-8"))
    else:
        raw = graph

    if not isinstance(raw, (list, tuple)):
        raise ValueError("Graph must be a JSON array or a sequence of node dictionaries.")

    nodes = [DagNode.from_mapping(item) for item in raw]
    validate_graph(nodes)
    return sorted(nodes, key=lambda n: n.node_id)


def parse_fewshot_file(path: Union[str, Path]) -> List[Tuple[str, List[Dict[str, object]]]]:
    """Parse a fewshot .txt file and extract (example_title, graph) for each Output block.

    Expects blocks like:
        Example1：
        ...
        Output:
        [ {...}, ... ]
    Supports both fullwidth (：) and halfwidth (:) colons after Example N and Output.
    """
    text = Path(path).read_text(encoding="utf-8")
    # Split into blocks by "Example" + digits + optional colon (fullwidth or halfwidth)
    block_pat = re.compile(r"(?=Example\d+[：:])", re.IGNORECASE)
    parts = block_pat.split(text)
    results: List[Tuple[str, List[Dict[str, object]]]] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Extract label: first line should be like "Example1：" or "Example2:"
        label_match = re.match(r"(Example\d+)[：:]*", part, re.IGNORECASE)
        if not label_match:
            continue
        title = label_match.group(1)

        # Find "Output" followed by colon (fullwidth or halfwidth)
        output_match = re.search(r"Output\s*[：:]\s*", part, re.IGNORECASE)
        if not output_match:
            continue
        after_output = part[output_match.end() :]
        # Find the first "[" and then the matching "]"
        start = after_output.find("[")
        if start == -1:
            continue
        depth = 0
        end = -1
        for i, ch in enumerate(after_output[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            continue
        json_str = after_output[start : end + 1]
        try:
            graph = json.loads(json_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(graph, list):
            continue
        results.append((title, graph))
    return results


def validate_graph(nodes: Sequence[DagNode]) -> None:
    ids = [n.node_id for n in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate NodeID values are not allowed.")

    id_set = set(ids)
    indeg: Dict[int, int] = {nid: 0 for nid in id_set}

    for node in nodes:
        for child in node.next_ids:
            if child not in id_set:
                raise ValueError(f"Node {node.node_id} points to unknown node {child}.")
            indeg[child] += 1

    queue = [nid for nid, d in indeg.items() if d == 0]
    seen = 0
    children: Dict[int, List[int]] = {n.node_id: list(n.next_ids) for n in nodes}
    while queue:
        current = queue.pop()
        seen += 1
        for child in children[current]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)

    if seen != len(nodes):
        raise ValueError("The graph is not a DAG; cycles are not supported.")


_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in text)


def wrap_text_latex(text: str, width_chars: int) -> str:
    words = text.split()
    if not words:
        return ""

    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width_chars:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return r" \\ ".join(latex_escape(line) for line in lines)


def compute_levels(nodes: Sequence[DagNode]) -> Dict[int, int]:
    node_map = {n.node_id: n for n in nodes}
    indeg = {n.node_id: 0 for n in nodes}
    for n in nodes:
        for c in n.next_ids:
            indeg[c] += 1

    roots = sorted([nid for nid, d in indeg.items() if d == 0])
    queue = list(roots)
    level = {nid: 0 for nid in roots}

    while queue:
        u = queue.pop(0)
        for v in node_map[u].next_ids:
            level[v] = max(level.get(v, 0), level[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    return level


def estimate_text_width_inches(question: str, top_k: str) -> float:
    longest_line_chars = max(
        len(f"Q: {question}"),
        len(f"Top-k: {top_k}"),
    )
    width = max(2.9, min(7.8, longest_line_chars / 11.0))
    return round(width, 2)


def build_node_body(node: DagNode, width_chars: int) -> str:
    q_wrapped = wrap_text_latex(node.question, width_chars)
    top_k = latex_escape(node.top_k)
    return (
        r"\textbf{Node %d} \\ "
        r"\textbf{Q:} %s \\ "
        r"\textbf{Top-k:} %s"
    ) % (node.node_id, q_wrapped, top_k)


def _action_style(action: str) -> str:
    """Return TikZ style name: 'retrieval' or 'reasoning'."""
    a = action.strip().lower()
    return "reasoning" if a == "reasoning" else "retrieval"


def level_order(nodes: Sequence[DagNode], levels: Mapping[int, int]) -> Dict[int, List[DagNode]]:
    by_level: Dict[int, List[DagNode]] = {}
    for n in nodes:
        by_level.setdefault(levels[n.node_id], []).append(n)
    for lv in by_level:
        by_level[lv].sort(key=lambda n: n.node_id)
    return by_level


def compute_node_widths_inches(nodes: Sequence[DagNode]) -> Dict[int, float]:
    return {
        n.node_id: estimate_text_width_inches(n.question, n.top_k)
        for n in nodes
    }


def compute_positions(
    nodes: Sequence[DagNode],
    levels: Mapping[int, int],
    width_in: Mapping[int, float],
    level_gap_cm: float = 3.4,
    node_gap_cm: float = 1.2,
) -> Dict[int, Tuple[float, float]]:
    by_level = level_order(nodes, levels)
    positions: Dict[int, Tuple[float, float]] = {}

    for lv, level_nodes in sorted(by_level.items()):
        widths_cm = [width_in[n.node_id] * 2.54 for n in level_nodes]
        total_width = sum(widths_cm) + node_gap_cm * max(0, len(level_nodes) - 1)
        left_edge = -total_width / 2.0
        cursor = left_edge
        y = -lv * level_gap_cm

        for node, w_cm in zip(level_nodes, widths_cm):
            x_center = cursor + w_cm / 2.0
            positions[node.node_id] = (round(x_center, 3), round(y, 3))
            cursor += w_cm + node_gap_cm

    return positions


def graph_to_tikz(
    graph: JsonLike,
    title: Optional[str] = None,
    level_gap_cm: float = 3.4,
    node_gap_cm: float = 1.2,
    standalone: bool = True,
) -> str:
    nodes = load_graph(graph)
    levels = compute_levels(nodes)
    widths_in = compute_node_widths_inches(nodes)
    positions = compute_positions(
        nodes,
        levels,
        width_in=widths_in,
        level_gap_cm=level_gap_cm,
        node_gap_cm=node_gap_cm,
    )

    title_tex = latex_escape(title) if title else None

    lines: List[str] = []
    if standalone:
        lines.extend(
            [
                r"\documentclass[tikz,border=8pt]{standalone}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage{lmodern}",
                r"\usepackage{textcomp}",
                r"\usepackage{tikz}",
                r"\usetikzlibrary{arrows.meta,calc,positioning}",
                "",
                r"\begin{document}",
            ]
        )

    max_level = max(levels.values()) if levels else 0
    level_gap_cm_val = level_gap_cm

    lines.extend(
        [
            r"\begin{tikzpicture}[",
            r"  x=1cm,",
            r"  y=1cm,",
            r"  >=Latex,",
            r"  line width=0.9pt,",
            r"  every node/.style={font=\small},",
            r"  dagbox/.style={",
            r"    draw,",
            r"    rounded corners=2pt,",
            r"    align=left,",
            r"    inner xsep=7pt,",
            r"    inner ysep=6pt,",
            r"    minimum height=1.3cm,",
            r"  },",
            r"  dagbox retrieval/.style={dagbox, fill=blue!12},",
            r"  dagbox reasoning/.style={dagbox, fill=orange!20},",
            r"  dagarrow/.style={->, line width=0.95pt},",
            r"]",
        ]
    )

    if title_tex:
        lines.append(rf"\node[font=\normalsize\bfseries] at (0,1.8) {{{title_tex}}};")

    for node in nodes:
        x, y = positions[node.node_id]
        width_this_in = widths_in[node.node_id]
        width_chars = max(24, min(70, int(math.floor(width_this_in * 11))))
        body = build_node_body(node, width_chars)
        style = _action_style(node.action)
        lines.append(
            rf"\node[dagbox {style}, text width={width_this_in:.2f}in] (N{node.node_id}) at ({x:.3f},{y:.3f}) {{{body}}};"
        )

    for node in nodes:
        for child in node.next_ids:
            lines.append(rf"\draw[dagarrow] (N{node.node_id}.south) -- (N{child}.north);")

    # Legend: Retrieval (blue) and Reasoning (orange)
    legend_y = -max_level * level_gap_cm_val - 1.4
    lines.append(r"\node[dagbox retrieval, minimum width=0.55cm, minimum height=0.35cm] at (-1.4," + f"{legend_y:.2f}" + r") {};")
    lines.append(rf"\node[anchor=west, font=\small] at (-1.1,{legend_y:.2f}) {{Retrieval}};")
    lines.append(r"\node[dagbox reasoning, minimum width=0.55cm, minimum height=0.35cm] at (0.4," + f"{legend_y:.2f}" + r") {};")
    lines.append(rf"\node[anchor=west, font=\small] at (0.7,{legend_y:.2f}) {{Reasoning}};")

    lines.append(r"\end{tikzpicture}")

    if standalone:
        lines.append(r"\end{document}")

    return "\n".join(lines)


def save_tex(
    graph: JsonLike,
    output_tex: Union[str, Path],
    title: Optional[str] = None,
    level_gap_cm: float = 3.4,
    node_gap_cm: float = 1.2,
    standalone: bool = True,
) -> Path:
    tex = graph_to_tikz(
        graph=graph,
        title=title,
        level_gap_cm=level_gap_cm,
        node_gap_cm=node_gap_cm,
        standalone=standalone,
    )
    out = Path(output_tex)
    out.write_text(tex, encoding="utf-8")
    return out


def compile_tex(tex_path: Path) -> bool:
    """Run pdflatex on the .tex file. Returns True if PDF was produced."""
    tex_path = tex_path.resolve()
    if not tex_path.is_file():
        return False
    out_dir = tex_path.parent
    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={out_dir}",
                str(tex_path.name),
            ],
            cwd=out_dir,
            capture_output=True,
            check=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return (out_dir / tex_path.with_suffix(".pdf").name).exists()


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a DAG-style QA decomposition graph to standalone TikZ/LaTeX."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to a JSON file (single graph) or a fewshot .txt file (multiple examples).",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to the .tex file to create (single graph), or directory for fewshot (one .tex per example).",
    )
    parser.add_argument(
        "--fewshot",
        action="store_true",
        help="Treat input as a fewshot .txt file; parse all Example N / Output blocks and write one .tex per example.",
    )
    parser.add_argument("--title", default=None, help="Optional title shown above the graph (single-graph mode).")
    parser.add_argument("--level-gap-cm", type=float, default=3.4, help="Vertical spacing between levels.")
    parser.add_argument("--node-gap-cm", type=float, default=1.2, help="Horizontal gap between nodes on the same level.")
    parser.add_argument(
        "--no-standalone",
        action="store_true",
        help="Emit only the tikzpicture instead of a full standalone document.",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Do not run pdflatex on the generated .tex file(s).",
    )
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    if args.fewshot:
        if not args.input or not args.output:
            parser.error("With --fewshot, both input (fewshot .txt) and output (directory) are required.")
        in_path = Path(args.input)
        out_path = Path(args.output)
        if not in_path.is_file():
            parser.error(f"Fewshot input is not a file: {in_path}")
        out_path.mkdir(parents=True, exist_ok=True)
        examples = parse_fewshot_file(in_path)
        if not examples:
            raise SystemExit("No Example/Output blocks found in the fewshot file.")
        for title, graph in examples:
            tex = graph_to_tikz(
                graph=graph,
                title=title,
                level_gap_cm=args.level_gap_cm,
                node_gap_cm=args.node_gap_cm,
                standalone=not args.no_standalone,
            )
            # example1.tex, example2.tex, ...
            base = title.lower().replace(" ", "_")
            out_file = out_path / f"{base}.tex"
            out_file.write_text(tex, encoding="utf-8")
            print(f"Wrote {out_file}")
            if not args.no_compile and not args.no_standalone:
                if compile_tex(out_file):
                    print(f"  → {out_file.with_suffix('.pdf')}")
                else:
                    print("  → pdflatex failed or not found")
        return

    if not args.input or not args.output:
        parser.error("Without --fewshot, both input (JSON) and output (.tex) are required.")
    out = save_tex(
        graph=Path(args.input),
        output_tex=Path(args.output),
        title=args.title,
        level_gap_cm=args.level_gap_cm,
        node_gap_cm=args.node_gap_cm,
        standalone=not args.no_standalone,
    )
    print(f"Wrote {out}")
    if not args.no_compile and not args.no_standalone:
        if compile_tex(out):
            print(f"  → {out.with_suffix('.pdf')}")
        else:
            print("  → pdflatex failed or not found")


if __name__ == "__main__":
    main()
