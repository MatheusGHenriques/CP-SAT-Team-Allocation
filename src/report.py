from __future__ import annotations

from typing import Dict, Any, List
from .metrics import fairness_report, explain_assignment


def _format_team_distribution(
    team_stats: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
    title: str,
) -> List[str]:
    lines = []
    if not team_stats:
        return lines
    if title:
        lines.append(title)
        lines.append('=' * 80)
    scores = meta['scores']
    prefs = meta['preferences']
    veterans = meta['veterans']
    metrics_cfg = meta.get('metrics', {}) or {}
    exclude_vets = bool(metrics_cfg.get('exclude_veterans_from_avg_score', False))
    avg_label = 'avg*' if exclude_vets else 'avg'
    for s in meta['subsystems']:
        info = team_stats[s]
        lines.append(
            f"[{s}]  size={info['size']}  veterans={info['veterans']}  "
            f"score={info['score']}  {avg_label}={info['avg_score']:.2f}"
        )
        ordered = sorted(info['members'], key=lambda p: scores[p], reverse=True)
        for p in ordered:
            rank = prefs[p][s]
            kind = 'Veteran' if p in veterans else 'Novice'
            lines.append(f"  - {p:<18} | {kind:<8} | Score {scores[p]:03d} | Rank: {rank}")
        lines.append('')
    return lines


def _format_frontier(frontier: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    if not frontier:
        return lines

    lines.append('PARETO FRONTIER')
    lines.append('=' * 80)
    lines.append('preference  fairness  score_balance  veteran_balance  penalties  floor')

    for i, point in enumerate(frontier, start=1):
        comps = point.get('components', {})
        lines.append(
            f"{i:>2}. {point.get('preference', 0):>10}  {point.get('fairness', 0):>8}  "
            f"{comps.get('score_balance', point.get('score_balance', 0)):>13}  "
            f"{comps.get('veteran_balance', point.get('veteran_balance', 0)):>15}  "
            f"{comps.get('penalties', point.get('penalties', 0)):>9}  "
            f"{point.get('preference_floor', '-') }"
        )

    lines.append('')
    return lines


def _format_knee_explanation(knee: Dict[str, Any]) -> List[str]:
    lines = []
    lines.append('')
    lines.append('Knee point: best trade-off between preference and fairness.')
    lines.append(f"At this point, preference={knee['preference']}, fairness={knee['fairness']}.")
    lines.append('Below is the team distribution for the knee point:')
    lines.append('')
    return lines


def _format_monte_carlo(mc: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if not mc:
        return lines

    settings = mc.get('settings', {})
    lines.append('MONTE CARLO ROBUSTNESS')
    lines.append('=' * 80)
    lines.append(f"Runs requested: {settings.get('runs', '-')}")
    lines.append(f"Successful runs: {mc.get('successful_runs', 0)}")
    lines.append(f"Failed runs: {mc.get('failed_runs', 0)}")
    lines.append(f"Success rate: {mc.get('success_rate', 0.0):.1%}")
    lines.append(
        f"Noise model: score ±{settings.get('score_jitter', 0)}, "
        f"preference ±{settings.get('preference_jitter', 0)} @ p={settings.get('preference_noise_probability', 0.0):.2f}"
    )
    lines.append(f"Mean objective: {mc.get('objective_mean', 0.0):.2f} ± {mc.get('objective_std', 0.0):.2f}")
    lines.append(f"Objective range: [{mc.get('objective_min', 0.0):.2f}, {mc.get('objective_max', 0.0):.2f}]")
    lines.append(f"Mean changed assignments: {mc.get('changed_mean', 0.0):.2f} ± {mc.get('changed_std', 0.0):.2f}")
    lines.append(f"Mean stability: {mc.get('stability_mean', 0.0):.4f} ± {mc.get('stability_std', 0.0):.4f}")
    lines.append(f"Stability range: [{mc.get('stability_min', 0.0):.4f}, {mc.get('stability_max', 0.0):.4f}]")
    lines.append('')

    lines.append('Interpretation: stability close to 1.0 means most people stay')
    lines.append('in the same team across perturbations.')
    lines.append('')

    if mc.get('top_sensitive'):
        lines.append('Most sensitive people (lowest baseline-team retention)')
        lines.append('-' * 80)
        for item in mc['top_sensitive']:
            lines.append(
                f"{item['person']:<18} | base={str(item['baseline_team']):<10} | "
                f"retention={item['retention']:.1%} | mode={str(item['most_common_team']):<10} | "
                f"entropy={item['normalized_entropy']:.3f}"
            )
        lines.append('')

    if mc.get('top_robust'):
        lines.append('Most robust people (highest baseline-team retention)')
        lines.append('-' * 80)
        for item in mc['top_robust']:
            lines.append(
                f"{item['person']:<18} | base={str(item['baseline_team']):<10} | "
                f"retention={item['retention']:.1%} | mode={str(item['most_common_team']):<10} | "
                f"entropy={item['normalized_entropy']:.3f}"
            )
        lines.append('')

    lines.append('Per-person retention table')
    lines.append('-' * 80)
    lines.append('person               base team   retention  mode team   entropy')
    person_stats = mc.get('person_stats', {})
    for person in sorted(person_stats.keys()):
        item = person_stats[person]
        lines.append(
            f"{person:<20} {str(item.get('baseline_team', '-')):<10} {item.get('retention', 0.0):>9.1%}  "
            f"{str(item.get('most_common_team', '-')):<10} {item.get('normalized_entropy', 0.0):>7.3f}"
        )

    lines.append('')
    return lines


def generate_report(solution: Dict[str, Any]) -> str:
    meta = solution['meta']
    prefs = meta['preferences']
    scores = meta['scores']
    veterans = meta['veterans']
    subsystems = meta['subsystems']
    team_sizes = meta['team_sizes']

    fair = fairness_report(solution)
    team_stats = fair['team_stats']
    stage_results = solution.get('stage_results', [])
    frontier = solution.get('pareto_frontier', [])
    pareto_knee = solution.get('pareto_knee')
    monte_carlo = solution.get('monte_carlo', {})

    lines: List[str] = []
    lines.append('ALLOCATION RESULT')
    lines.append('=' * 80)
    lines.append(f"Solver status: {solution['status_name']}")
    lines.append(f"Objective (primary stage): {solution['objective']:.2f}")
    lines.append('')

    if stage_results:
        lines.append('Optimization stages:')
        for st in stage_results:
            direction = 'max' if st['direction'] == 'max' else 'min'
            lines.append(f"  {st['stage']:<16} | {direction:<3} | value = {st['value']}")
    lines.append('')

    lines.extend(
        _format_team_distribution(
            team_stats, meta,
            'OPTIMAL DISTRIBUTION (baseline)'
        )
    )

    lines.append('FAIRNESS')
    lines.append('=' * 80)
    lines.append(f"1st choice: {fair['first_choice']}/{len(meta['people'])}")
    lines.append(f"Top-2:    {fair['top2']}/{len(meta['people'])}")
    lines.append(f"Top-3:    {fair['top3']}/{len(meta['people'])}")
    lines.append(f"Jain index (scores): {fair['jain_scores']:.4f}")
    lines.append(f"Jain index (sizes):  {fair['jain_sizes']:.4f}")
    lines.append(f"Score range: {fair['score_range']}")
    lines.append(f"Size range: {fair['size_range']}")

    avg_label = 'avg (novices only)' if fair.get('exclude_veterans') else 'avg'
    lines.append(f"Jain index ({avg_label}):  {fair['jain_avg_scores']:.4f}")
    lines.append(f"{avg_label.capitalize()} range: {fair['avg_score_range']}")
    lines.append('')
    lines.append('Jain index ranges from 0 to 1 (1 = perfectly fair).')
    lines.append('Values above 0.95 indicate good balance across teams.')
    lines.append('')

    if frontier:
        lines.extend(_format_frontier(frontier))

        if pareto_knee:
            lines.extend(_format_knee_explanation(pareto_knee))
            knee_ts = pareto_knee.get('team_stats')
            knee_meta = pareto_knee['meta']
            lines.extend(
                _format_team_distribution(
                    knee_ts, knee_meta,
                    ''
                )
            )

    lines.append('ASSIGNMENT EXPLANATION')
    lines.append('=' * 80)
    for p in meta['people']:
        team = solution['assignment'][p]
        explanation = explain_assignment(p, team, team_stats, meta)
        lines.append(f"{p:<18} -> {team:<10} | {explanation}")

    lines.append('')

    if monte_carlo:
        lines.extend(_format_monte_carlo(monte_carlo))
        rep_ts = monte_carlo.get('representative_team_stats')
        if rep_ts:
            lines.append(
                'The distribution below shows where each person most often ends up'
            )
            lines.append('across the Monte Carlo runs (mode assignment).')
            lines.append('Note: team sizes may differ from constraints since each')
            lines.append("person's mode is computed independently.")
            lines.append('')
            lines.extend(
                _format_team_distribution(
                    rep_ts, meta,
                    'REPRESENTATIVE DISTRIBUTION (Monte Carlo mode)'
                )
            )

    lines.append('CURRENT CONFIGURATION')
    lines.append('=' * 80)
    lines.append(f"Sizes: {team_sizes}")
    lines.append(f"Min veterans per team: {meta['constraints'].get('min_veterans_per_team', 1)}")
    lines.append(f"Optimization mode: {meta.get('optimization', {}).get('mode', 'lexicographic')}")
    if stage_results:
        stages_str = ' -> '.join(st['stage'] for st in stage_results)
        lines.append(f"Stages: {stages_str}")
    lines.append(f"Preference weights: {meta['weights']['preference']}")
    lines.append(f"High score bonus: {meta['weights'].get('high_performance_bonus', {})}")
    lines.append(f"Penalties: {meta['weights'].get('penalties', {})}")

    if frontier:
        lines.append('')
        lines.append(f"Pareto frontier points: {len(frontier)}")

    if monte_carlo:
        s = monte_carlo.get('settings', {})
        lines.append(f"Monte Carlo: {s.get('runs', '?')} runs, "
                     f"{monte_carlo.get('success_rate', 0.0):.1%} success rate")

    return '\n'.join(lines)
