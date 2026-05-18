from __future__ import annotations

import logging
from collections import Counter
from copy import deepcopy
from math import log2
from random import Random
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

import tqdm

from .metrics import compute_team_stats
from .solver import solve_lexicographic

logger = logging.getLogger(__name__)


def compare_assignments(base_assignment: Dict[str, str], other_assignment: Dict[str, str]) -> Dict[str, Any]:
    common_people = set(base_assignment) & set(other_assignment)
    changed = [p for p in common_people if base_assignment[p] != other_assignment[p]]
    changed_count = len(changed)
    stability = 1.0 - (changed_count / max(1, len(common_people)))
    return {
        'changed_count': changed_count,
        'changed_people': changed,
        'stability': stability,
    }


def _safe_stdev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def _entropy_from_counter(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in counter.values():
        p = count / total
        ent -= p * log2(p)
    return ent


def _perturb_data(
    data: Dict[str, Any],
    rng: Random,
    score_jitter: int,
    preference_jitter: int,
    preference_noise_probability: float,
) -> Dict[str, Any]:
    perturbed = deepcopy(data)
    for _, payload in perturbed['people'].items():
        score = int(payload['score'])
        if score_jitter > 0:
            score += rng.randint(-score_jitter, score_jitter)
            score = max(0, score)
        payload['score'] = score

        prefs = dict(payload['preferences'])
        if preference_jitter > 0 and preference_noise_probability > 0:
            for subsystem, rank in list(prefs.items()):
                if rng.random() < preference_noise_probability:
                    delta = rng.randint(-preference_jitter, preference_jitter)
                    prefs[subsystem] = max(1, min(5, int(rank) + int(delta)))
        payload['preferences'] = prefs

    return perturbed


def _mc_run_task(args):
    data, seed, score_jitter, preference_jitter, prob, time_limit = args
    rng = Random(seed)
    perturbed = _perturb_data(data, rng, score_jitter, preference_jitter, prob)
    solution = solve_lexicographic(perturbed, time_limit=time_limit)
    if solution is None:
        return None
    return solution['assignment'], float(solution['objective'])


def run_monte_carlo_robustness(
    data: Dict[str, Any],
    baseline_solution: Optional[Dict[str, Any]] = None,
    runs: Optional[int] = None,
    score_jitter: Optional[int] = None,
    preference_jitter: Optional[int] = None,
    preference_noise_probability: Optional[float] = None,
    time_limit_per_run: Optional[float] = None,
    seed: Optional[int] = None,
    parallel: bool = True,
    quiet: bool = False,
) -> Dict[str, Any]:
    mc_cfg = data.get('monte_carlo', {}) or {}
    settings = {
        'runs': int(runs if runs is not None else mc_cfg.get('runs', 50)),
        'score_jitter': int(score_jitter if score_jitter is not None else mc_cfg.get('score_jitter', 4)),
        'preference_jitter': int(preference_jitter if preference_jitter is not None else mc_cfg.get('preference_jitter', 1)),
        'preference_noise_probability': float(
            preference_noise_probability if preference_noise_probability is not None else mc_cfg.get('preference_noise_probability', 0.05)
        ),
        'time_limit_per_run': float(time_limit_per_run if time_limit_per_run is not None else mc_cfg.get('time_limit_per_run', 5.0)),
        'seed': seed if seed is not None else mc_cfg.get('seed', None),
    }

    if settings['runs'] <= 0:
        raise ValueError('Monte Carlo runs must be positive.')
    if settings['time_limit_per_run'] <= 0:
        raise ValueError('Monte Carlo time_limit_per_run must be positive.')
    if not (0.0 <= settings['preference_noise_probability'] <= 1.0):
        raise ValueError('Monte Carlo preference_noise_probability must be in [0, 1].')

    if baseline_solution is None:
        logger.info('Computing baseline solution for Monte Carlo')
        baseline_solution = solve_lexicographic(data)

    if baseline_solution is None:
        logger.error('Baseline solution failed, Monte Carlo cannot proceed')
        return {
            'settings': settings,
            'successful_runs': 0,
            'failed_runs': settings['runs'],
            'success_rate': 0.0,
            'objective_mean': 0.0,
            'objective_std': 0.0,
            'objective_min': 0.0,
            'objective_max': 0.0,
            'changed_mean': 0.0,
            'changed_std': 0.0,
            'stability_mean': 0.0,
            'stability_std': 0.0,
            'stability_min': 0.0,
            'stability_max': 0.0,
            'person_stats': {},
            'top_sensitive': [],
            'top_robust': [],
            'runs_detail': [],
        }

    baseline_assignment = baseline_solution['assignment']
    people = list(data['people'].keys())

    # Prepare task arguments for each run
    rng = Random(settings['seed'])
    task_args = []
    for idx in range(settings['runs']):
        run_seed = rng.randint(0, 2**31)
        task_args.append((
            data, run_seed,
            settings['score_jitter'], settings['preference_jitter'],
            settings['preference_noise_probability'],
            settings['time_limit_per_run'],
        ))

    raw_results: List[Dict[str, Any] | None] = []

    if parallel and settings['runs'] > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        logger.info('Running %d Monte Carlo trials in parallel', settings['runs'])
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_mc_run_task, a): a for a in task_args}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures),
                                    desc='Monte Carlo', disable=quiet):
                raw_results.append(future.result())
    else:
        logger.info('Running %d Monte Carlo trials sequentially', settings['runs'])
        for args in tqdm.tqdm(task_args, desc='Monte Carlo', disable=quiet):
            raw_results.append(_mc_run_task(args))

    objective_values: List[float] = []
    changed_counts: List[int] = []
    stabilities: List[float] = []
    runs_detail: List[Dict[str, Any]] = []

    team_counts = {p: Counter() for p in people}
    same_team_counts = {p: 0 for p in people}

    successful_runs = 0
    failed_runs = 0

    for idx, result in enumerate(raw_results):
        run_num = idx + 1
        if result is None:
            failed_runs += 1
            continue

        assignment, objective = result
        successful_runs += 1
        cmp = compare_assignments(baseline_assignment, assignment)

        objective_values.append(objective)
        changed_counts.append(int(cmp['changed_count']))
        stabilities.append(float(cmp['stability']))

        for person, team in assignment.items():
            team_counts[person][team] += 1
            if baseline_assignment.get(person) == team:
                same_team_counts[person] += 1

        runs_detail.append({
            'run': run_num,
            'objective': objective,
            'changed_count': int(cmp['changed_count']),
            'stability': float(cmp['stability']),
        })

    logger.info('Monte Carlo: %d successful, %d failed out of %d',
                successful_runs, failed_runs, settings['runs'])

    person_stats: Dict[str, Dict[str, Any]] = {}
    for person in people:
        counts = team_counts[person]
        baseline_team = baseline_assignment.get(person)
        retention = (same_team_counts[person] / successful_runs) if successful_runs else 0.0
        most_common_team = counts.most_common(1)[0][0] if counts else None
        most_common_team_freq = (counts.most_common(1)[0][1] / successful_runs) if counts and successful_runs else 0.0
        entropy = _entropy_from_counter(counts)
        max_entropy = log2(len(counts)) if len(counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        person_stats[person] = {
            'baseline_team': baseline_team,
            'retention': retention,
            'most_common_team': most_common_team,
            'most_common_team_frequency': most_common_team_freq,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'assignment_counts': dict(counts),
            'samples': sum(counts.values()),
        }

    sorted_people = sorted(
        person_stats.items(),
        key=lambda item: (item[1]['retention'], item[1]['normalized_entropy'])
    )
    top_sensitive = [
        {'person': p, **stats}
        for p, stats in sorted_people[:5]
    ]
    top_robust = [
        {'person': p, **stats}
        for p, stats in sorted(person_stats.items(), key=lambda item: (-item[1]['retention'], item[1]['normalized_entropy']))[:5]
    ]

    rep_assignment: Dict[str, str] = {}
    for person, counter in team_counts.items():
        mc = counter.most_common(1)
        if mc:
            rep_assignment[person] = mc[0][0]

    rep_team_members: Dict[str, List[str]] = {s: [] for s in data['team_sizes']}
    for person, team in rep_assignment.items():
        if team in rep_team_members:
            rep_team_members[team].append(person)

    rep_solution = {
        'assignment': rep_assignment,
        'team_members': rep_team_members,
        'meta': baseline_solution['meta'],
    }
    rep_team_stats = compute_team_stats(rep_solution)

    return {
        'settings': settings,
        'baseline_assignment': baseline_assignment,
        'successful_runs': successful_runs,
        'failed_runs': failed_runs,
        'success_rate': successful_runs / settings['runs'],
        'objective_mean': mean(objective_values) if objective_values else 0.0,
        'objective_std': _safe_stdev(objective_values),
        'objective_min': min(objective_values) if objective_values else 0.0,
        'objective_max': max(objective_values) if objective_values else 0.0,
        'changed_mean': mean(changed_counts) if changed_counts else 0.0,
        'changed_std': _safe_stdev(changed_counts),
        'stability_mean': mean(stabilities) if stabilities else 0.0,
        'stability_std': _safe_stdev(stabilities),
        'stability_min': min(stabilities) if stabilities else 0.0,
        'stability_max': max(stabilities) if stabilities else 0.0,
        'person_stats': person_stats,
        'top_sensitive': top_sensitive,
        'top_robust': top_robust,
        'runs_detail': runs_detail,
        'representative_assignment': rep_assignment,
        'representative_team_members': rep_team_members,
        'representative_team_stats': rep_team_stats,
    }
