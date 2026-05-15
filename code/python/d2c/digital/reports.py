"""Human-readable reports for digital D2C experiments."""

from __future__ import annotations

from typing import Mapping


def format_symbolic_induction_report(result: Mapping[str, object]) -> str:
    summary = result["summary"]
    lines = [
        "Symbolic Induction Report",
        "=========================",
        f"Experiment: {result['experiment_name']}",
        f"Variant:    {result['variant']}",
        f"Gap:        {result['gap']}",
        f"Seeds:      {result['seed_count']}",
        "",
        "Metrics",
        "-------",
        f"Accuracy:   {summary['accuracy']:.3f} ({summary['correct']}/{summary['count']})",
        f"Mean Score: {summary['mean_score']:.6f}",
        "",
        "Memory",
        "------",
        f"Gamma:           {result['memory_config']['gamma']}",
        f"Weights:         {result['memory_config']['w']}",
        f"D_eff:           {result['memory_diagnostics']['deff']:.3f}",
        f"Stability Ratio: {result['memory_diagnostics']['stability_ratio']:.3f}",
    ]
    if result.get("baseline"):
        baseline = result["baseline"]
        lines.extend([
            "",
            "Baseline",
            "--------",
            f"Window:   {baseline['window']}",
            f"Accuracy: {baseline['accuracy']:.3f} ({baseline['correct']}/{baseline['count']})",
        ])
    return "\n".join(lines)
