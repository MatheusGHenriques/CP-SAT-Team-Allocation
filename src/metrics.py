from __future__ import annotations

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def jain_index(values: List[int]) -> float:
    values = [max(0, int(v)) for v in values]
    s = sum(values)
    ss = sum(v * v for v in values)
    if ss == 0:
        return 1.0
    return (s * s) / (len(values) * ss)


def compute_team_stats(solution: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    meta = solution['meta']
    scores = meta['scores']
    prefs = meta['preferences']
    veterans = meta['veterans']
    subsystems = meta['subsystems']
    metrics_cfg = meta.get('metrics', {}) or {}
    exclude_vets = bool(metrics_cfg.get('exclude_veterans_from_avg_score', False))

    team_stats: Dict[str, Dict[str, Any]] = {}

    for s in subsystems:
        members = solution['team_members'][s]
        size = len(members)
        score = sum(scores[p] for p in members)
        vet_count = sum(1 for p in members if p in veterans)

        if exclude_vets:
            novices = [p for p in members if p not in veterans]
            novice_score = sum(scores[p] for p in novices)
            novice_count = len(novices)
            avg_score = novice_score / novice_count if novice_count else 0
        else:
            avg_score = score / size if size else 0

        ranks = [prefs[p][s] for p in members]
        avg_rank = sum(ranks) / len(ranks) if ranks else 0

        team_stats[s] = {
            'members': members,
            'size': size,
            'score': score,
            'avg_score': avg_score,
            'veterans': vet_count,
            'avg_rank': avg_rank,
        }

    logger.debug('Team stats computed for %d subsystems', len(subsystems))
    return team_stats


def explain_assignment(person: str, team: str, team_stats: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> str:
    rank = meta['preferences'][person][team]
    score = meta['scores'][person]
    info = team_stats[team]

    ordinal = {1: '1st choice', 2: '2nd choice', 3: '3rd choice', 4: '4th choice', 5: '5th choice'}
    parts = [ordinal.get(rank, f'{rank}th choice')]

    if person in meta['veterans']:
        parts.append('veteran')
    else:
        parts.append('novice')

    if team in ('OBC', 'ADCS'):
        parts.append('priority subsystem for high scores')

    if score >= 90:
        parts.append('high-score contributor')

    if info['veterans'] >= 1:
        parts.append(f"{info['veterans']} veteran(s) on team")

    return ', '.join(parts)


def fairness_report(solution: Dict[str, Any]) -> Dict[str, Any]:
    meta = solution['meta']
    prefs = meta['preferences']
    subsystems = meta['subsystems']

    team_stats = compute_team_stats(solution)
    team_scores = [team_stats[s]['score'] for s in subsystems]
    team_sizes = [team_stats[s]['size'] for s in subsystems]
    avg_team_scores = [int(round(team_stats[s]['avg_score'])) for s in subsystems]

    first_choice = 0
    top2 = 0
    top3 = 0

    for p, s in solution['assignment'].items():
        r = prefs[p][s]
        first_choice += int(r == 1)
        top2 += int(r <= 2)
        top3 += int(r <= 3)

    logger.info('Fairness: %d/%d first choices, Jain scores=%.4f',
                first_choice, len(meta['people']), jain_index(team_scores))

    result = {
        'team_scores': team_scores,
        'team_sizes': team_sizes,
        'jain_scores': jain_index(team_scores),
        'jain_sizes': jain_index(team_sizes),
        'score_range': max(team_scores) - min(team_scores),
        'size_range': max(team_sizes) - min(team_sizes),
        'first_choice': first_choice,
        'top2': top2,
        'top3': top3,
        'team_stats': team_stats,
        'avg_team_scores': avg_team_scores,
        'jain_avg_scores': jain_index(avg_team_scores),
        'avg_score_range': max(avg_team_scores) - min(avg_team_scores),
    }

    metrics_cfg = meta.get('metrics', {}) or {}
    result['exclude_veterans'] = bool(metrics_cfg.get('exclude_veterans_from_avg_score', False))
    return result
