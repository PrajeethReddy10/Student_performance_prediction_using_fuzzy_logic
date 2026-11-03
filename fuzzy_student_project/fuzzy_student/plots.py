"""Figures (requires matplotlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import sweep
from .engine import MamdaniFIS

COLORS = {"low": "#c0392b", "poor": "#c0392b", "weak": "#c0392b",
          "medium": "#e08e0b", "average": "#e08e0b",
          "high": "#1e8449", "good": "#1e8449", "excellent": "#1f4e9c"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path: Path, plt) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def membership_functions(fis: MamdaniFIS, path: Path) -> Path:
    plt = _plt()
    vs = list(fis.inputs.values()) + [fis.output]
    cols = 3
    rows = int(np.ceil(len(vs) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(11, 3.2 * rows))
    axs = np.atleast_1d(axs).ravel()
    for ax, v in zip(axs, vs):
        for t in v.terms:
            ax.plot(v.universe, v.curve(t), lw=2, label=t, color=COLORS.get(t))
        ax.set_title(f"{v.name} ({v.unit})")
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("membership")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axs[len(vs):]:
        ax.axis("off")
    return _save(fig, path, plt)


def surface(fis: MamdaniFIS, xv: str, yv: str, fixed: dict, path: Path) -> Path:
    plt = _plt()
    xs = np.linspace(fis.inputs[xv].lo, fis.inputs[xv].hi, 30)
    ys = np.linspace(fis.inputs[yv].lo, fis.inputs[yv].hi, 30)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[fis.infer(**{xv: x, yv: y, **fixed}).score for x in xs] for y in ys])
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel(xv)
    ax.set_ylabel(yv)
    ax.set_zlabel("predicted score")
    ax.view_init(28, -125)
    ax.set_title("Performance surface (" + ", ".join(f"{k}={v:g}" for k, v in fixed.items()) + ")")
    return _save(fig, path, plt)


def aggregation(fis: MamdaniFIS, student: dict, path: Path) -> Path:
    """Aggregated output set and its defuzzified value for one student."""
    plt = _plt()
    r = fis.infer(**student)
    o = fis.output
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for t in o.terms:
        ax.plot(o.universe, o.curve(t), "--", lw=1, color=COLORS.get(t), label=t)
    ax.fill_between(o.universe, r.aggregated, color="#1f4e9c", alpha=0.35, label="aggregated")
    ax.axvline(r.score, color="k", lw=1.5)
    ax.text(r.score + 1, 0.95, f"{fis.defuzz} = {r.score:.1f}")
    ax.set_xlabel("performance score")
    ax.set_ylabel("membership")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    return _save(fig, path, plt)


def sensitivity(fis: MamdaniFIS, base: dict, path: Path) -> Path:
    plt = _plt()
    n = len(fis.inputs)
    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3), sharey=True)
    for ax, var in zip(axs, fis.inputs):
        xs, ys = sweep(fis, base, var)
        ax.plot(xs, ys, lw=2, color="#1f4e9c")
        ax.set_xlabel(var)
        ax.grid(alpha=0.3)
        for c in (25, 50, 75):
            ax.axhline(c, color="grey", lw=0.6, ls=":")
    axs[0].set_ylabel("predicted score")
    fig.suptitle("One-at-a-time sensitivity (others at " + ", ".join(f"{k}={v:g}" for k, v in base.items()) + ")", fontsize=9)
    return _save(fig, path, plt)


def predicted_vs_actual(y: np.ndarray, pred: np.ndarray, title: str, path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.scatter(y, pred, s=14, alpha=0.6, color="#1f4e9c")
    ax.plot([0, 100], [0, 100], "k--", lw=0.8)
    ax.set_xlabel("true score (synthetic)")
    ax.set_ylabel("predicted score")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    return _save(fig, path, plt)


def confusion_matrix(cm: np.ndarray, labels: list[str], title: str, path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    return _save(fig, path, plt)


def ablation_bars(rows: list[dict], path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = [r["name"] for r in rows][::-1]
    ax.barh(names, [r["cal_rmse"] for r in rows][::-1], color="#1f4e9c")
    ax.set_xlabel("calibrated RMSE on test split (lower is better)")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path, plt)


def rule_usage_bars(usage: list[dict], path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(range(len(usage)), [u["top"] for u in usage], color="#1f4e9c")
    ax.set_xlabel("rule index")
    ax.set_ylabel("times strongest rule")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path, plt)


def tuning_history(history: list[float], path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(history, lw=2, color="#1f4e9c")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("train correlation r")
    ax.grid(alpha=0.3)
    return _save(fig, path, plt)
