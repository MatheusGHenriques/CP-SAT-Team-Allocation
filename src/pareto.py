from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import tqdm
from ortools.sat.python import cp_model

from .model import build_model
from .solver import solve_lexicographic
from .metrics import compute_team_stats
from .solution import FEASIBLE_STATUS, build_assignment_dict

logger = logging.getLogger(__name__)


def _unique_signature(assignment: Dict[str, str]) -> tuple:
    return tuple(sorted(assignment.items()))


def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ap, af = a['preference'], a['fairness']
    bp, bf = b['preference'], b['fairness']
    return (ap >= bp and af <= bf) and (ap > bp or af < bf)


def _filter_nondominated(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nondominated: List[Dict[str, Any]] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if _dominates(q, p):
                dominated = True
                break
        if not dominated:
            nondominated.append(p)

    nondominated.sort(key=lambda d: (-d['preference'], d['fairness']))
    return nondominated


def solve_pareto_point(
    data: Dict[str, Any],
    preference_floor: Optional[int] = None,
    time_limit: Optional[float] = None,
) -> Dict[str, Any] | None:
    model, x, meta, components = build_model(data)

    pareto_cfg = meta.get('pareto', {})
    fairness_weights = pareto_cfg.get('fairness_weights', {}) or {}
    w_score = int(fairness_weights.get('score_balance', 1))
    w_veteran = int(fairness_weights.get('veteran_balance', 1))
    w_penalty = int(fairness_weights.get('penalties', 1))

    if preference_floor is not None:
        model.Add(components['preference'] >= int(preference_floor))

    fairness_expr = (
        w_score * components['score_balance']
        + w_veteran * components['veteran_balance']
        + w_penalty * components['penalties']
    )

    model.Minimize(fairness_expr)

    solver = cp_model.CpSolver()
    if time_limit is None:
        time_limit = float(meta['constraints'].get('max_time_seconds', 20))
    solver.parameters.max_time_in_seconds = float(time_limit)

    status = solver.Solve(model)
    if status not in FEASIBLE_STATUS:
        return None

    solution = build_assignment_dict(solver, x, meta)
    solution['status'] = status
    solution['status_name'] = solver.StatusName(status)
    solution['preference'] = int(round(solver.Value(components['preference'])))
    solution['score_balance'] = int(round(solver.Value(components['score_balance'])))
    solution['veteran_balance'] = int(round(solver.Value(components['veteran_balance'])))
    solution['penalties'] = int(round(solver.Value(components['penalties'])))
    solution['fairness'] = int(round(solver.Value(fairness_expr)))
    solution['team_stats'] = compute_team_stats(solution)
    solution['components'] = {
        'preference': solution['preference'],
        'score_balance': solution['score_balance'],
        'veteran_balance': solution['veteran_balance'],
        'penalties': solution['penalties'],
        'fairness': solution['fairness'],
    }
    return solution


def _solve_pareto_point_task(args):
    data, preference_floor, time_limit = args
    result = solve_pareto_point(data, preference_floor=preference_floor, time_limit=time_limit)
    if result is not None:
        result.pop('solver', None)
        result.pop('x', None)
    return result


def generate_pareto_frontier(
    data: Dict[str, Any],
    max_points: Optional[int] = None,
    min_preference_ratio: Optional[float] = None,
    time_limit_per_point: Optional[float] = None,
    parallel: bool = True,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    pareto_cfg = data.get('pareto', {}) or {}
    if max_points is None:
        max_points = int(pareto_cfg.get('points', 6))
    if min_preference_ratio is None:
        min_preference_ratio = float(pareto_cfg.get('min_preference_ratio', 0.9))
    if time_limit_per_point is None:
        time_limit_per_point = float(pareto_cfg.get('time_limit_per_point', 8))

    logger.info('Computing baseline for Pareto frontier')
    baseline = solve_lexicographic(data, time_limit=time_limit_per_point)
    if baseline is None or not baseline.get('stage_results'):
        logger.error('Baseline solution failed, cannot generate Pareto frontier')
        return []

    best_preference = int(baseline['stage_results'][0]['value'])
    min_preference = max(0, int(best_preference * float(min_preference_ratio)))
    logger.info('Preference range: %d to %d (%d points)', best_preference, min_preference, max_points)

    if max_points < 2:
        max_points = 2

    thresholds: List[int] = []
    if max_points == 2:
        thresholds = [best_preference, min_preference]
    else:
        span = best_preference - min_preference
        for i in range(max_points):
            if max_points == 1:
                thr = best_preference
            else:
                thr = int(round(best_preference - (span * i) / (max_points - 1)))
            thresholds.append(thr)

    thresholds = sorted(set(max(0, t) for t in thresholds), reverse=True)

    task_args = [(data, t, time_limit_per_point) for t in thresholds]
    raw_points: List[Dict[str, Any]] = []

    if parallel and len(task_args) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        logger.info('Solving %d Pareto points in parallel', len(task_args))
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_solve_pareto_point_task, a): a for a in task_args}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures),
                                    desc='Pareto points', disable=quiet):
                point = future.result()
                if point is not None:
                    raw_points.append(point)
    else:
        logger.info('Solving %d Pareto points sequentially', len(task_args))
        for args in tqdm.tqdm(task_args, desc='Pareto points', disable=quiet):
            point = _solve_pareto_point_task(args)
            if point is not None:
                raw_points.append(point)

    seen = set()
    points: List[Dict[str, Any]] = []
    for point in raw_points:
        sig = _unique_signature(point['assignment'])
        if sig in seen:
            continue
        seen.add(sig)
        for t in thresholds:
            if abs(point['preference'] - t) <= 1:
                point['preference_floor'] = t
                break
        else:
            point['preference_floor'] = thresholds[thresholds.index(min(thresholds, key=lambda x: abs(x - point['preference'])))]
        points.append(point)

    frontier = _filter_nondominated(points)
    logger.info('Pareto frontier: %d non-dominated points', len(frontier))
    return frontier


def find_knee_point(frontier: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not frontier:
        return None
    if len(frontier) <= 2:
        return frontier[-1]

    fair_vals = [float(p['fairness']) for p in frontier]
    pref_vals = [float(p['preference']) for p in frontier]

    f_min, f_max = min(fair_vals), max(fair_vals)
    p_min, p_max = min(pref_vals), max(pref_vals)

    f_range = f_max - f_min if f_max != f_min else 1.0
    p_range = p_max - p_min if p_max != p_min else 1.0

    first, last = frontier[0], frontier[-1]
    x1 = (float(first['fairness']) - f_min) / f_range
    y1 = (float(first['preference']) - p_min) / p_range
    x2 = (float(last['fairness']) - f_min) / f_range
    y2 = (float(last['preference']) - p_min) / p_range

    line_len = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    if line_len < 1e-9:
        return frontier[len(frontier) // 2]

    max_dist = -1.0
    knee = frontier[0]

    for point in frontier:
        x0 = (float(point['fairness']) - f_min) / f_range
        y0 = (float(point['preference']) - p_min) / p_range
        dist = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / line_len
        if dist > max_dist:
            max_dist = dist
            knee = point

    logger.debug('Knee point at index %d with distance %.4f', frontier.index(knee), max_dist)
    return knee
