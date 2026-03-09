"""TeX → PDF compilation for learning-curve artifacts.

``compile_all`` compiles every ``.tex`` file under ``CONFIG_TRAIN_CURVES_DIR/tex/``
to PDF.  ``compile_one`` compiles a single file.  All intermediate artifacts are
created in ``CONFIG_EPHEMERAL_TMPDIR`` (Container disk) and final PDFs are copied
to the persistent ``curves/pdf/`` directory.

Compilation failures are **non-fatal**: they are logged at WARNING level and
recorded in ``manifests/compile_status.json``.  Training never crashes due to
TeX compilation errors.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


def compile_all(config: "Config") -> None:
    """Compile all ``.tex`` files in the curves tex directory to PDF."""
    tex_dir = os.path.join(config.TRAIN_CURVES_DIR, "tex")
    if not os.path.isdir(tex_dir):
        logger.warning("TeX directory does not exist: %s", tex_dir)
        return

    tex_files = sorted(
        f for f in os.listdir(tex_dir) if f.endswith(".tex")
    )
    if not tex_files:
        logger.debug("No .tex files found in %s", tex_dir)
        return

    statuses: List[Dict] = []
    for tex_file in tex_files:
        tex_path = os.path.join(tex_dir, tex_file)
        status = compile_one(tex_path, config)
        statuses.append(status)

    _write_compile_status(config, statuses)


def compile_one(tex_path: str, config: "Config") -> Dict:
    """Compile a single ``.tex`` file to PDF.

    Scratch artifacts go into ``CONFIG_EPHEMERAL_TMPDIR``; the final PDF
    is copied to ``CONFIG_TRAIN_CURVES_DIR/pdf/``.

    Returns a status dict for ``compile_status.json``.
    """
    pdf_dir = os.path.join(config.TRAIN_CURVES_DIR, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)

    tex_filename = os.path.basename(tex_path)
    pdf_filename = tex_filename.replace(".tex", ".pdf")
    timeout_sec = config.TRAIN_CURVES_PDFLATEX_TIMEOUT_SEC

    tmpdir = tempfile.mkdtemp(
        dir=config.EPHEMERAL_TMPDIR,
        prefix="tex_compile_",
    )

    status: Dict = {
        "file": tex_filename,
        "success": False,
        "error": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    try:
        tmp_tex = os.path.join(tmpdir, tex_filename)
        shutil.copy2(tex_path, tmp_tex)

        cmd = _build_compile_cmd(config, tmpdir, tex_filename)

        result = subprocess.run(
            cmd,
            cwd=tmpdir,
            timeout=timeout_sec,
            capture_output=True,
            text=True,
        )

        tmp_pdf = os.path.join(tmpdir, pdf_filename)
        if result.returncode == 0 and os.path.isfile(tmp_pdf):
            dest = os.path.join(pdf_dir, pdf_filename)
            shutil.copy2(tmp_pdf, dest)
            status["success"] = True
            logger.debug("Compiled %s → %s", tex_filename, dest)
        else:
            msg = (result.stderr or result.stdout or "")[-500:]
            status["error"] = f"returncode={result.returncode}: {msg}"
            logger.warning(
                "TeX compilation failed for %s (rc=%d)",
                tex_filename, result.returncode,
            )

    except subprocess.TimeoutExpired:
        status["error"] = f"Timed out after {timeout_sec}s"
        logger.warning("TeX compilation timed out for %s", tex_filename)

    except Exception as exc:
        status["error"] = str(exc)
        logger.warning("TeX compilation error for %s: %s", tex_filename, exc)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return status


def _find_executable(name: str) -> str | None:
    """Return full path for name, or None if not in PATH."""
    return shutil.which(name)


def _build_compile_cmd(
    config: "Config",
    tmpdir: str,
    tex_filename: str,
) -> List[str]:
    """Build the compilation command based on config."""
    engine = config.TRAIN_CURVES_LATEX_ENGINE

    if config.TRAIN_CURVES_LATEXMK:
        latexmk_cmd = _find_executable("latexmk")
        if latexmk_cmd is not None:
            return [
                latexmk_cmd,
                "-pdf",
                f"-{engine}" if engine != "pdflatex" else "-pdflatex",
                f"-output-directory={tmpdir}",
                tex_filename,
            ]
        # Fallback to pdflatex when latexmk is not in PATH
        logger.debug("latexmk not found in PATH, using %s directly", engine)

    engine_cmd = _find_executable(engine) or engine
    return [
        engine_cmd,
        "-interaction=nonstopmode",
        f"-output-directory={tmpdir}",
        tex_filename,
    ]


def _write_compile_status(config: "Config", statuses: List[Dict]) -> None:
    """Write per-file compilation status to the manifests directory."""
    manifests_dir = os.path.join(config.TRAIN_CURVES_DIR, "manifests")
    os.makedirs(manifests_dir, exist_ok=True)

    status_path = os.path.join(manifests_dir, "compile_status.json")
    with open(status_path, "w") as f:
        json.dump(statuses, f, indent=2)


# ---------------------------------------------------------------------------
# CLI entrypoint for: python -m src.training.tex_compile --config .config --all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    from src.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Compile learning curve TeX files to PDF")
    parser.add_argument("--config", default=".config", help="Path to Kconfig .config file")
    parser.add_argument("--all", action="store_true", help="Compile all .tex files in curves/tex")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    config = load_config(args.config)
    if args.all:
        compile_all(config)
    else:
        logger.warning("Use --all to compile all curves")
        sys.exit(0)
