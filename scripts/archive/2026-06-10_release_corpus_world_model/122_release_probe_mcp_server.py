from __future__ import annotations

"""
122_release_probe_mcp_server.py

Consume:
- Repo-local probe scripts `scripts/120_curve_corpus_audit.py`
- Repo-local probe scripts `scripts/121_shape_family_audit.py`
- Their emitted artifacts under `outputs/120_curve_corpus_audit/` and
  `outputs/121_shape_family_audit/`

Produce:
- A local stdio MCP server for Claude Desktop with tools to:
  - inspect current probe summaries
  - read key probe artifacts
  - rerun the two probe scripts

Expected runtime:
- server startup: < 3 s
- `probe_status` / `read_probe_artifact`: < 1 s
- `run_curve_corpus_audit`: ~5-20 s depending on external xlsx availability
- `run_shape_family_audit`: ~20-120 s depending on corpus size
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_120 = REPO_ROOT / "outputs" / "120_curve_corpus_audit"
OUTPUTS_121 = REPO_ROOT / "outputs" / "121_shape_family_audit"
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

ARTIFACTS = {
    "corpus_summary_md": OUTPUTS_120 / "summary.md",
    "corpus_summary_json": OUTPUTS_120 / "corpus_summary.json",
    "corpus_curve_stats_csv": OUTPUTS_120 / "curve_stats.csv",
    "corpus_group_summary_csv": OUTPUTS_120 / "group_summary.csv",
    "shape_summary_md": OUTPUTS_121 / "summary.md",
    "shape_family_summary_csv": OUTPUTS_121 / "family_summary.csv",
    "shape_best_family_summary_csv": OUTPUTS_121 / "best_family_summary.csv",
    "shape_per_curve_metrics_csv": OUTPUTS_121 / "per_curve_family_metrics.csv",
}


def project_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def run_repo_script(script_name: str) -> dict[str, object]:
    script_path = REPO_ROOT / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    command = [project_python(), str(script_path)]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def read_text_artifact(path: Path) -> str:
    if not path.exists():
        return f"[missing] {path}"
    return path.read_text(encoding="utf-8")


def truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    return f"{head}\n\n[truncated: showing first {limit} chars of {len(text)}]"


def safe_json_load(path: Path) -> dict[str, object] | list[object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_status_markdown() -> str:
    corpus_json = safe_json_load(ARTIFACTS["corpus_summary_json"])
    shape_summary = read_text_artifact(ARTIFACTS["shape_summary_md"])
    lines = [
        "# Release Probe Status",
        "",
        f"- repo_root: `{REPO_ROOT}`",
        f"- project_python: `{project_python()}`",
        "",
    ]
    if isinstance(corpus_json, dict):
        lines.append("## Corpus Audit Headlines")
        lines.append("")
        for key, payload in corpus_json.items():
            if not isinstance(payload, dict):
                continue
            lines.append(
                f"- `{key}`: curves={payload.get('n_curves')}, "
                f"obs={payload.get('n_observations')}, "
                f"groups={payload.get('n_groups')}, "
                f"monotone_clean_fraction={payload.get('monotone_clean_fraction')}"
            )
        lines.append("")
    lines.append("## Shape Family Audit Summary")
    lines.append("")
    lines.append(truncate(shape_summary, limit=8000))
    lines.append("")
    return "\n".join(lines)


mcp = FastMCP("release-probes")


@mcp.tool()
def probe_status() -> str:
    """Return a compact markdown snapshot of the current probe artifacts."""
    return build_status_markdown()


@mcp.tool()
def read_probe_artifact(
    artifact_name: Annotated[
        Literal[
            "corpus_summary_md",
            "corpus_summary_json",
            "corpus_curve_stats_csv",
            "corpus_group_summary_csv",
            "shape_summary_md",
            "shape_family_summary_csv",
            "shape_best_family_summary_csv",
            "shape_per_curve_metrics_csv",
        ],
        Field(description="Named artifact emitted by the release probe scripts."),
    ],
) -> str:
    """Read one of the known probe artifacts from the repo outputs directory."""
    path = ARTIFACTS[artifact_name]
    return truncate(read_text_artifact(path))


@mcp.tool()
def run_curve_corpus_audit() -> str:
    """Rerun script 120 and return the execution result plus output paths."""
    result = run_repo_script("120_curve_corpus_audit.py")
    lines = [
        "# Curve Corpus Audit Run",
        "",
        f"- returncode: `{result['returncode']}`",
        f"- summary: `{ARTIFACTS['corpus_summary_md']}`",
        f"- json: `{ARTIFACTS['corpus_summary_json']}`",
        "",
        "## stdout",
        "",
        "```text",
        str(result["stdout"]),
        "```",
        "",
        "## stderr",
        "",
        "```text",
        str(result["stderr"]),
        "```",
    ]
    return "\n".join(lines)


@mcp.tool()
def run_shape_family_audit() -> str:
    """Rerun script 121 and return the execution result plus output paths."""
    result = run_repo_script("121_shape_family_audit.py")
    lines = [
        "# Shape Family Audit Run",
        "",
        f"- returncode: `{result['returncode']}`",
        f"- summary: `{ARTIFACTS['shape_summary_md']}`",
        f"- family_summary: `{ARTIFACTS['shape_family_summary_csv']}`",
        f"- best_family_summary: `{ARTIFACTS['shape_best_family_summary_csv']}`",
        "",
        "## stdout",
        "",
        "```text",
        str(result["stdout"]),
        "```",
        "",
        "## stderr",
        "",
        "```text",
        str(result["stderr"]),
        "```",
    ]
    return "\n".join(lines)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
