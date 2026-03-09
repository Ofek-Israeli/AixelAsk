"""Learning-curve management: TSV data files, TeX templates, and metric manifest.

``CurvesManager`` maintains incremental TSV data files and renders Jinja2-
templated PGFPlots ``.tex`` files for all tracked training metrics.  It is
driven by ``CurvesCallback`` (in ``train_stats.py``) which calls
``update_tsv`` and ``generate_tex`` on the configured cadence.

Directory layout (under ``CONFIG_TRAIN_CURVES_DIR``)::

    data/          — one TSV per metric (step \\t value)
    tex/           — one .tex per metric + family overlays
    pdf/           — compiled PDFs (written by tex_compile.py)
    manifests/     — metrics_manifest.json, compile_status.json
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from jinja2 import Template

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric catalogue
# ---------------------------------------------------------------------------

_STEP_METRICS = [
    {"metric": "reward_mean", "ylabel": "Reward", "family": "reward"},
    {"metric": "reward_std", "ylabel": "Reward Std", "family": None},
    {"metric": "correctness_rate", "ylabel": "Correctness", "family": "correctness"},
    {"metric": "validity_rate", "ylabel": "Validity", "family": "validity"},
    {"metric": "depth_mean", "ylabel": "Depth", "family": "depth"},
    {"metric": "invalid_rate", "ylabel": "Invalid Rate", "family": None},
    {"metric": "train_total_loss", "ylabel": "Loss", "family": None},
    {"metric": "train_policy_loss", "ylabel": "Policy Loss", "family": None},
    {"metric": "train_kl_loss", "ylabel": "KL Loss", "family": None},
    {"metric": "grad_norm", "ylabel": "Grad Norm", "family": None},
    {"metric": "lr", "ylabel": "Learning Rate", "family": None},
    {"metric": "response_len_mean", "ylabel": "Response Length", "family": None},
    {"metric": "parse_success_rate", "ylabel": "Parse Success", "family": None},
    {"metric": "advantage_mean", "ylabel": "Advantage", "family": None},
    {"metric": "group_reward_std_mean", "ylabel": "Group Reward Std", "family": None},
]

_EVAL_METRICS = [
    {"metric": "eval_reward_mean", "ylabel": "Reward", "family": "reward"},
    {"metric": "eval_correctness_rate", "ylabel": "Correctness", "family": "correctness"},
    {"metric": "eval_validity_rate", "ylabel": "Validity", "family": "validity"},
    {"metric": "eval_depth_mean", "ylabel": "Depth", "family": "depth"},
]

_FAMILIES = {
    "reward": {
        "title": "Reward Mean",
        "ylabel": "Reward",
        "train_metric": "reward_mean",
        "eval_metric": "eval_reward_mean",
    },
    "correctness": {
        "title": "Correctness Rate",
        "ylabel": "Correctness",
        "train_metric": "correctness_rate",
        "eval_metric": "eval_correctness_rate",
    },
    "validity": {
        "title": "Validity Rate",
        "ylabel": "Validity",
        "train_metric": "validity_rate",
        "eval_metric": "eval_validity_rate",
    },
    "depth": {
        "title": "Depth Mean",
        "ylabel": "Depth",
        "train_metric": "depth_mean",
        "eval_metric": "eval_depth_mean",
    },
}

# ---------------------------------------------------------------------------
# Jinja2 TeX templates
# ---------------------------------------------------------------------------

_SINGLE_TEX_TEMPLATE = Template(r"""\documentclass[tikz,border=5pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\begin{document}
\begin{tikzpicture}
\begin{axis}[
  title={ {{- title -}} },
  xlabel={Step},
  ylabel={ {{- ylabel -}} },
  width=14cm,
  height=8cm,
  grid=major,
  line width=0.8pt,
]
\addplot[blue, mark=none] table[x=step, y=value, col sep=tab] { {{- data_path -}} };
\end{axis}
\end{tikzpicture}
\end{document}
""")

_FAMILY_TEX_TEMPLATE = Template(r"""\documentclass[tikz,border=5pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\begin{document}
\begin{tikzpicture}
\begin{axis}[
  title={ {{- title -}} },
  xlabel={Step},
  ylabel={ {{- ylabel -}} },
  width=14cm,
  height=8cm,
  grid=major,
  legend pos=south east,
  line width=0.8pt,
]
\addplot[blue, mark=none] table[x=step, y=value, col sep=tab] { {{- train_data_path -}} };
\addlegendentry{Train}
\addplot[red, mark=*, mark size=1.5pt] table[x=step, y=value, col sep=tab] { {{- eval_data_path -}} };
\addlegendentry{Eval}
\end{axis}
\end{tikzpicture}
\end{document}
""")


# ---------------------------------------------------------------------------
# CurvesManager
# ---------------------------------------------------------------------------

class CurvesManager:
    """Manages TSV data files, TeX templates, and the metrics manifest."""

    def __init__(self, config: "Config") -> None:
        self._config = config
        self._curves_dir = config.TRAIN_CURVES_DIR
        self._data_dir = os.path.join(self._curves_dir, "data")
        self._tex_dir = os.path.join(self._curves_dir, "tex")
        self._pdf_dir = os.path.join(self._curves_dir, "pdf")
        self._manifests_dir = os.path.join(self._curves_dir, "manifests")
        self._keep_last_n = config.TRAIN_CURVES_KEEP_LAST_N_POINTS

        self._all_metrics: List[Dict[str, Any]] = []
        for m in _STEP_METRICS:
            self._all_metrics.append({**m, "source": "step"})
        for m in _EVAL_METRICS:
            self._all_metrics.append({**m, "source": "eval"})

    # -------------------------------------------------------------------
    # init_curves_dir
    # -------------------------------------------------------------------

    def init_curves_dir(self) -> None:
        """Create directory structure, write metrics_manifest.json, and
        generate initial (empty-data) TeX files."""
        for d in (self._data_dir, self._tex_dir, self._pdf_dir, self._manifests_dir):
            os.makedirs(d, exist_ok=True)

        for entry in self._all_metrics:
            tsv_path = os.path.join(self._data_dir, f"{entry['metric']}.tsv")
            if not os.path.exists(tsv_path):
                with open(tsv_path, "w") as f:
                    f.write("step\tvalue\n")

        manifest_path = os.path.join(self._manifests_dir, "metrics_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(self._all_metrics, f, indent=2)

        self.generate_tex()
        logger.info("Curves directory initialised: %s", self._curves_dir)

    # -------------------------------------------------------------------
    # update_tsv
    # -------------------------------------------------------------------

    def update_tsv(self, metric_name: str, step: int, value: float) -> None:
        """Append a data point to the TSV file for *metric_name*.

        If ``TRAIN_CURVES_KEEP_LAST_N_POINTS > 0``, the file is trimmed
        to the last N rows after appending.
        """
        tsv_path = os.path.join(self._data_dir, f"{metric_name}.tsv")
        os.makedirs(os.path.dirname(tsv_path), exist_ok=True)
        if not os.path.exists(tsv_path):
            with open(tsv_path, "w") as f:
                f.write("step\tvalue\n")

        with open(tsv_path, "a") as f:
            f.write(f"{step}\t{value}\n")

        if self._keep_last_n > 0:
            self._trim_tsv(tsv_path, self._keep_last_n)

    def _trim_tsv(self, tsv_path: str, keep_n: int) -> None:
        """Keep only the header + last *keep_n* data rows."""
        with open(tsv_path, "r") as f:
            lines = f.readlines()
        if len(lines) <= keep_n + 1:
            return
        trimmed = [lines[0]] + lines[-(keep_n):]
        with open(tsv_path, "w") as f:
            f.writelines(trimmed)

    # -------------------------------------------------------------------
    # generate_tex
    # -------------------------------------------------------------------

    def generate_tex(self) -> None:
        """Render all Jinja2 TeX templates with absolute data paths."""
        for entry in self._all_metrics:
            metric = entry["metric"]
            title = metric.replace("_", " ").title()
            ylabel = entry["ylabel"]
            data_path = os.path.abspath(
                os.path.join(self._data_dir, f"{metric}.tsv")
            )
            tex_content = _SINGLE_TEX_TEMPLATE.render(
                title=title,
                ylabel=ylabel,
                data_path=data_path,
            )
            tex_path = os.path.join(self._tex_dir, f"{metric}.tex")
            with open(tex_path, "w") as f:
                f.write(tex_content)

        for family_key, family_info in _FAMILIES.items():
            train_data_path = os.path.abspath(
                os.path.join(self._data_dir, f"{family_info['train_metric']}.tsv")
            )
            eval_data_path = os.path.abspath(
                os.path.join(self._data_dir, f"{family_info['eval_metric']}.tsv")
            )
            tex_content = _FAMILY_TEX_TEMPLATE.render(
                title=family_info["title"],
                ylabel=family_info["ylabel"],
                train_data_path=train_data_path,
                eval_data_path=eval_data_path,
            )
            tex_path = os.path.join(self._tex_dir, f"{family_key}_family.tex")
            with open(tex_path, "w") as f:
                f.write(tex_content)

    # -------------------------------------------------------------------
    # get_metrics_manifest
    # -------------------------------------------------------------------

    def get_metrics_manifest(self) -> List[Dict[str, Any]]:
        """Return the list of tracked metrics with metadata."""
        return list(self._all_metrics)
