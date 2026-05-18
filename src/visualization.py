from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('ggplot')


def _ensure_parent_dir(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finalize(fig: plt.Figure, output_path: str | Path) -> Path:
    path = _ensure_parent_dir(output_path)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    logger.info('Saved chart to %s', path)
    return path


def generate_pareto_frontier_scatter(
    frontier: Sequence[Dict[str, Any]],
    output_path: str | Path = 'pareto_frontier.png',
    knee_point: Optional[Dict[str, Any]] = None,
) -> Path | None:
    if not frontier:
        return None

    points = sorted(
        frontier,
        key=lambda p: (float(p.get('fairness', 0.0)), -float(p.get('preference', 0.0)))
    )

    x = [float(p.get('fairness', 0.0)) for p in points]
    y = [float(p.get('preference', 0.0)) for p in points]
    labels = [str(p.get('preference_floor', i + 1)) for i, p in enumerate(points)]

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
    ax.scatter(x, y, c=colors, s=90, edgecolors='#333333', linewidths=0.5, zorder=3)
    ax.plot(x, y, linestyle='--', linewidth=1.0, alpha=0.4, color='#555555', zorder=2)

    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords='offset points',
                    xytext=(7, 7), fontsize=9, alpha=0.85)

    if knee_point is not None:
        kx = float(knee_point.get('fairness', 0.0))
        ky = float(knee_point.get('preference', 0.0))
        ax.scatter(kx, ky, c='#d62728', s=220, marker='*',
                   edgecolors='black', linewidths=0.6, zorder=5, label='Knee point')
        ax.annotate('KNEE', (kx, ky), textcoords='offset points',
                    xytext=(10, -14), fontsize=11, fontweight='bold', color='#d62728')
        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')

    ax.set_title('Pareto Frontier: Preference vs Fairness', fontsize=13, fontweight='bold')
    ax.set_xlabel('Fairness cost (lower is better)', fontsize=11)
    ax.set_ylabel('Preference score (higher is better)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')

    return _finalize(fig, output_path)


def generate_team_composition_heatmap(
    team_stats: Dict[str, Dict[str, Any]],
    output_path: str | Path = 'team_composition_heatmap.png',
) -> Path | None:
    if not team_stats:
        return None

    teams = list(team_stats.keys())
    columns = ['size', 'veterans', 'score', 'avg_score', 'avg_rank']
    raw = np.array([
        [float(team_stats[t].get(col, 0.0)) for col in columns]
        for t in teams
    ], dtype=float)

    normalized = raw.copy()
    for j in range(normalized.shape[1]):
        col = normalized[:, j]
        cmin = float(np.min(col))
        cmax = float(np.max(col))
        if abs(cmax - cmin) < 1e-9:
            normalized[:, j] = 0.0
        else:
            normalized[:, j] = (col - cmin) / (cmax - cmin)

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.75 * len(teams))))
    cmap = plt.cm.Blues
    im = ax.imshow(normalized, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, fontsize=10)
    ax.set_yticks(np.arange(len(teams)))
    ax.set_yticklabels(teams, fontsize=10)

    for i in range(len(teams)):
        for j in range(len(columns)):
            value = raw[i, j]
            is_float = columns[j] in {'avg_score', 'avg_rank'}
            text = f'{value:.2f}' if is_float else f'{int(round(value))}'
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color='white' if normalized[i, j] > 0.6 else '#333333')

    ax.set_title('Team Composition', fontsize=13, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Subsystem', fontsize=11)
    fig.colorbar(im, ax=ax, label='Normalized', shrink=0.85)

    return _finalize(fig, output_path)


def generate_assignment_stability_heatmap(
    mc_results: Dict[str, Any],
    people: Sequence[str],
    subsystems: Sequence[str],
    output_path: str | Path = 'stability_heatmap.png',
) -> Path | None:
    if not mc_results:
        return None

    person_stats = mc_results.get('person_stats', {}) or {}
    if not person_stats:
        return None

    matrix: List[List[float]] = []
    for p in people:
        counts = person_stats.get(p, {}).get('assignment_counts', {}) or {}
        samples = max(1, int(person_stats.get(p, {}).get('samples', sum(counts.values()) or 1)))
        row = [float(counts.get(s, 0)) / samples for s in subsystems]
        matrix.append(row)

    matrix_np = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(people))))
    cmap = plt.cm.viridis
    im = ax.imshow(matrix_np, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(subsystems)))
    ax.set_xticklabels(subsystems, fontsize=10)
    ax.set_yticks(np.arange(len(people)))
    ax.set_yticklabels(people, fontsize=9)

    threshold = 0.55
    for i in range(len(people)):
        for j in range(len(subsystems)):
            val = matrix_np[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if val > threshold else '#333333')

    ax.set_title('Monte Carlo: Assignment Stability', fontsize=13, fontweight='bold')
    ax.set_xlabel('Subsystem', fontsize=11)
    ax.set_ylabel('Person', fontsize=11)
    fig.colorbar(im, ax=ax, label='Assignment probability', shrink=0.85)

    return _finalize(fig, output_path)


def generate_assignment_confidence_barplot(
    mc_results: Dict[str, Any],
    output_path: str | Path = 'assignment_confidence.png',
) -> Path | None:
    if not mc_results:
        return None

    person_stats = mc_results.get('person_stats', {}) or {}
    if not person_stats:
        return None

    rows = []
    for person, stats in person_stats.items():
        confidence = float(stats.get('retention', 0.0))
        entropy = float(stats.get('normalized_entropy', 0.0))
        rows.append((person, confidence, entropy))

    rows.sort(key=lambda t: (t[1], -t[2]))
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.35 * len(names))))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    bars = ax.barh(names, values, color=colors, edgecolor='#333333', linewidth=0.3)

    ax.set_xlim(0, 1)
    ax.set_xlabel('Baseline retention', fontsize=11)
    ax.set_title('Monte Carlo: Assignment Confidence', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)

    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=8)

    return _finalize(fig, output_path)
