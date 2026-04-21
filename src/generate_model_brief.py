"""Create a model-review brief from model metrics.

Usage:
    python src/generate_model_brief.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    metrics_path = REPORTS / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}. Run `python src/train_model.py` first."
        )

    metrics = json.loads(metrics_path.read_text())
    sorted_metrics = sorted(metrics, key=lambda r: r["roc_auc"], reverse=True)

    lines = [
        "# Model Brief: Heart Disease Risk Modeling",
        "",
        "You are a healthcare-aware ML reviewer.",
        "Given the results below, produce:",
        "1) model comparison summary,",
        "2) deployment recommendation,",
        "3) risk/ethics caveats,",
        "4) next 3 experiments to improve recall.",
        "",
        "## Metrics",
    ]

    for row in sorted_metrics:
        lines.extend(
            [
                f"- Model: {row['model']}",
                f"  - ROC-AUC: {row['roc_auc']:.4f}",
                f"  - PR-AUC: {row['pr_auc']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "## Constraints",
            "- This is a screening support prototype, not diagnosis.",
            "- Minimize false negatives where possible.",
            "- Keep recommendations explainable for clinicians.",
        ]
    )

    out_path = REPORTS / "model_brief.md"
    out_path.write_text("\n".join(lines))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
