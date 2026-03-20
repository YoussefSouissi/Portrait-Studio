"""
Visualisation utilities for training analysis and evaluation results.
  - plot_training_history: loss curve, LR schedule, loss histogram
  - plot_fid_results: global FID and per-domain FID bar charts
  - plot_dataset_distribution: dataset attribute distributions
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available — visualisations disabled")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# -------------------------------------------------
# Training History
# -------------------------------------------------

def plot_training_history(
    history_path: Path,
    output_dir: Path = None,
) -> Optional[Path]:
    """Plot loss curve, LR schedule, loss histogram, and summary statistics."""

    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available")
        return None

    history_path = Path(history_path)
    if not history_path.exists():
        logger.warning(f"History file not found: {history_path}")
        return None

    with open(history_path, "r") as f:
        history = json.load(f)

    output_dir = Path(output_dir) if output_dir else history_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

    steps   = history.get("steps", [])
    losses  = history.get("losses", [])
    # FIX: use .get() — key may be absent if no LR was logged
    lrs     = history.get("learning_rates", [])

    # ── Loss curve ────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    if losses:
        plot_steps = steps if len(steps) == len(losses) else range(len(losses))
        ax.plot(plot_steps, losses, "b-", linewidth=1.2, label="Loss")
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("MSE Loss", fontsize=10)
        ax.set_title("Training Loss", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Training Loss", fontweight="bold")

    # ── LR schedule ──────────────────────────────────────────────────────────
    ax = axes[0, 1]
    if lrs:
        plot_steps = steps if len(steps) == len(lrs) else range(len(lrs))
        ax.plot(plot_steps, lrs, "g-", linewidth=1.2, label="LR")
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("Learning Rate", fontsize=10)
        ax.set_title("Learning Rate Schedule", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No LR data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Learning Rate Schedule", fontweight="bold")

    # ── Loss histogram ────────────────────────────────────────────────────────
    ax = axes[1, 0]
    if len(losses) > 10:
        ax.hist(losses, bins=min(50, len(losses) // 2), edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel("Loss Value", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title("Loss Distribution", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "Not enough data for histogram", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title("Loss Distribution", fontweight="bold")

    # ── Summary text ──────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    stats_text = "Training Statistics\n" + "─" * 28 + "\n"
    stats_text += f"Total steps logged: {len(steps)}\n"

    if losses:
        arr = np.array(losses)
        stats_text += f"Mean loss :  {arr.mean():.6f}\n"
        stats_text += f"Min  loss :  {arr.min():.6f}\n"
        stats_text += f"Max  loss :  {arr.max():.6f}\n"
        stats_text += f"Std  loss :  {arr.std():.6f}\n"
    else:
        stats_text += "(no loss values)\n"

    if lrs:
        stats_text += f"Final LR  :  {lrs[-1]:.3e}\n"

    ax.text(0.1, 0.5, stats_text, fontsize=11, family="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved training curves to {output_path}")
    return output_path


# -------------------------------------------------
# FID Results
# -------------------------------------------------

def plot_fid_results(
    fid_global_path: Path = None,
    fid_per_domain_path: Path = None,
    output_dir: Path = None,
) -> Optional[Path]:
    """
    Plot FID global bar + FID per-domain bar chart.

    FIX: shows informative placeholder text instead of blank axes when:
      - fid_global.json is missing / not passed
      - fid_per_domain.json is missing, not passed, or contains an empty dict {}
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("FID Score Analysis", fontsize=16, fontweight="bold")

    # ── Left: Global FID ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Global FID", fontweight="bold")

    fid_global_path = Path(fid_global_path) if fid_global_path else None
    if fid_global_path and fid_global_path.exists():
        with open(fid_global_path, "r") as f:
            data = json.load(f)
        fid_val = data.get("fid_global")
        n_real  = data.get("num_real", "?")
        n_gen   = data.get("num_generated", "?")

        if fid_val is not None and not np.isnan(fid_val):
            ax.bar(["FID Global"], [fid_val], color="steelblue", alpha=0.8)
            ax.set_ylim(0, max(50, fid_val * 1.25))
            ax.text(0, fid_val + max(0.5, fid_val * 0.02),
                    f"{fid_val:.2f}", ha="center", va="bottom",
                    fontweight="bold", fontsize=12)
            ax.set_ylabel("FID Score (lower = better)", fontsize=10)
            ax.set_xlabel(f"real={n_real} | gen={n_gen}", fontsize=9)
        else:
            # FIX: show message instead of blank axes
            ax.text(0.5, 0.5, "FID = NaN\n(insufficient generated images)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="firebrick",
                    bbox=dict(boxstyle="round", facecolor="mistyrose", alpha=0.7))
    else:
        # FIX: file missing → explain clearly
        ax.text(0.5, 0.5, "fid_global.json not found\n(FID was not computed)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    ax.grid(True, alpha=0.3, axis="y")

    # ── Right: Per-Domain FID ─────────────────────────────────────────────────
    ax = axes[1]
    ax.set_title("FID per Domain", fontweight="bold")

    fid_per_domain_path = Path(fid_per_domain_path) if fid_per_domain_path else None
    if fid_per_domain_path and fid_per_domain_path.exists():
        with open(fid_per_domain_path, "r") as f:
            data = json.load(f)

        domains    = list(data.keys())
        fid_scores = list(data.values())

        # FIX: guard for empty dict → blank chart with weird auto-scaled axes
        if domains:
            colors = plt.cm.Set2(np.linspace(0, 1, len(domains)))
            bars   = ax.bar(domains, fid_scores, color=colors, alpha=0.8, edgecolor="black")

            ax.set_ylabel("FID Score (lower = better)", fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")

            # Value labels on bars
            for bar, score in zip(bars, fid_scores):
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + max(0.3, h * 0.02),
                    f"{score:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )

            # FIX: prevent domain label clipping with rotation_mode and bottom margin
            plt.setp(
                ax.xaxis.get_majorticklabels(),
                rotation=40, ha="right", rotation_mode="anchor",
            )
            fig.subplots_adjust(bottom=0.18)
        else:
            # FIX: empty dict → show explanation instead of blank chart
            ax.text(
                0.5, 0.5,
                "No per-domain FID computed\n\n"
                "Reason: too few generated images per domain\n"
                f"(need ≥50 per domain for reliable FID;\n"
                "increase num_samples_per_prompt in the eval cell)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="firebrick",
                bbox=dict(boxstyle="round", facecolor="mistyrose", alpha=0.7),
            )
    else:
        ax.text(
            0.5, 0.5,
            "fid_per_domain.json not found\n(per-domain FID was not computed)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color="gray",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
        )

    ax.grid(True, alpha=0.3, axis="y")

    output_dir = Path(output_dir) if output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "fid_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved FID visualisation to {output_path}")
    return output_path


# -------------------------------------------------
# Dataset Distribution
# -------------------------------------------------

def plot_dataset_distribution(
    stats_path: Path,
    output_dir: Path = None,
) -> Optional[Path]:
    """Plot class/domain distribution, balance ratios, and quality issues."""

    if not MATPLOTLIB_AVAILABLE:
        return None

    stats_path = Path(stats_path)
    if not stats_path.exists():
        logger.warning(f"Stats file not found: {stats_path}")
        return None

    with open(stats_path, "r") as f:
        stats = json.load(f)

    output_dir = Path(output_dir) if output_dir else stats_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Distribution Analysis", fontsize=16, fontweight="bold")

    # ── Class distribution ────────────────────────────────────────────────────
    ax = axes[0, 0]
    classes = stats.get("classes", {})
    if isinstance(classes, dict) and classes:
        ax.barh(list(classes.keys()), list(classes.values()), color="skyblue", alpha=0.8)
        ax.set_xlabel("Count", fontsize=10)
        ax.set_title("Class Distribution", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
    else:
        ax.text(0.5, 0.5, "No class data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Class Distribution", fontweight="bold")

    # ── Domain pie chart ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    domains = stats.get("domains", {})
    if isinstance(domains, dict) and domains:
        colors = plt.cm.Set2(np.linspace(0, 1, len(domains)))
        ax.pie(domains.values(), labels=domains.keys(),
               autopct="%1.1f%%", colors=colors, startangle=90)
        ax.set_title("Domain Distribution", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No domain data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Domain Distribution", fontweight="bold")

    # ── Balance statistics ────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.axis("off")

    stats_text = "Balance Statistics\n" + "─" * 26 + "\n\n"
    if "domain_balance_ratio" in stats:
        stats_text += f"Domain imbalance: {stats['domain_balance_ratio']:.2f}×\n"
    if "class_balance_ratio" in stats:
        stats_text += f"Class  imbalance: {stats['class_balance_ratio']:.2f}×\n"
    stats_text += f"\nTotal images: {stats.get('total_images', 'N/A')}\n"

    ps = stats.get("prompt_stats", {})
    if ps:
        stats_text += f"Unique prompts: {ps.get('unique_prompts', 'N/A')}\n"
        avg = ps.get("avg_prompt_length") or ps.get("avg_length", 0)
        stats_text += f"Avg prompt length: {avg:.1f} words\n"

    ax.text(0.1, 0.5, stats_text, fontsize=11, family="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))

    # ── Data quality ──────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    quality = stats.get("quality_issues", {})
    n_corrupt  = len(quality.get("corrupted", []))
    n_missing  = len(quality.get("missing",   []))

    q_text = "Data Quality\n" + "─" * 20 + "\n\n"
    q_text += f"Corrupted images: {n_corrupt}\n"
    q_text += f"Missing files:    {n_missing}\n"

    if quality.get("corrupted"):
        q_text += "\nSample corrupted:\n"
        for item in quality["corrupted"][:3]:
            q_text += f"  • {Path(item.get('path', 'unknown')).name}\n"
        if n_corrupt > 3:
            q_text += f"  … and {n_corrupt - 3} more\n"

    color = "lightyellow" if n_corrupt == 0 and n_missing == 0 else "mistyrose"
    ax.text(0.1, 0.5, q_text, fontsize=10, family="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.6))

    plt.tight_layout()
    output_path = output_dir / "dataset_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved dataset distribution plot to {output_path}")
    return output_path


