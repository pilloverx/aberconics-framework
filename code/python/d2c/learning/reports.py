"""Human-readable reporting helpers for Phase 4 learning analysis."""

from __future__ import annotations

from .analysis import TraceLearningAnalysis


def format_learning_analysis_report(analysis: TraceLearningAnalysis) -> str:
    """Format an offline learning analysis into a human-readable text report."""

    lines = [
        "Learning Analysis Report",
        "========================",
        f"Experiment: {analysis.experiment_name}",
        f"Source:     {analysis.source}",
        f"Windows:    {analysis.window_count} (size={analysis.window_size}, stride={analysis.stride})",
        "",
        "Summary Metrics",
        "---------------",
        f"Mean Prediction Error (L2): {analysis.mean_prediction_error_l2:.6f}",
        "",
        "Stability & Kernels",
        "-------------------",
        f"Rescale Events:             {analysis.stability_rescale_count} / {analysis.window_count}",
        f"Mean Stability Ratio:       {analysis.mean_stability_ratio_after:.4f}",
        "",
    ]

    if analysis.notes:
        lines.append("Notes")
        lines.append("-----")
        for note in analysis.notes:
            lines.append(f"- {note}")
        lines.append("")

    if analysis.windows:
        lines.append("Window Highlights")
        lines.append("-----------------")
        # Show first, middle, and last window summaries if many
        show_indices = [0]
        if len(analysis.windows) > 2:
            show_indices.append(len(analysis.windows) // 2)
        if len(analysis.windows) > 1:
            show_indices.append(len(analysis.windows) - 1)

        # Ensure unique and sorted
        show_indices = sorted(list(set(show_indices)))

        for idx in show_indices:
            win = analysis.windows[idx]
            lines.append(f"Window {idx} (IC {win.ic_index}, indices {win.start_idx}-{win.end_idx}):")
            lines.append(f"  Reward:     {win.reward:+.6f}")
            err_str = f"{win.prediction.error_l2:.6f}" if win.prediction.error_l2 is not None else "N/A"
            lines.append(f"  Pred Error: {err_str}")
            if win.update_proposal:
                p = win.update_proposal
                lines.append(f"  Stability:  {p.stability_ratio_before:.4f} -> {p.stability_ratio_after:.4f} "
                             f"{'(RESCALED)' if p.stability_rescaled else ''}")
                # Optional: show weight deltas if they are small in number
                if len(p.old_weights) <= 8:
                    deltas = [new - old for old, new in zip(p.old_weights, p.proposed_weights)]
                    lines.append(f"  W-Deltas:   {[f'{d:+.4f}' for d in deltas]}")
            if win.notes:
                for note in win.notes:
                    lines.append(f"  Note: {note}")
            lines.append("")

    return "\n".join(lines)
