from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
from ortools.sat.python import cp_model

from .model import build_model
from .solution import FEASIBLE_STATUS, build_assignment_dict

logger = logging.getLogger(__name__)


def _stage_direction(stage_name: str) -> str:
    if stage_name in {'preference'}:
        return 'max'
    return 'min'


def solve_lexicographic(data: Dict[str, Any], time_limit: Optional[float] = None) -> Dict[str, Any] | None:
    model, x, meta, components = build_model(data)

    opt_cfg = meta.get('optimization', {})
    stages = opt_cfg.get('stages', ['preference', 'score_balance', 'veteran_balance', 'penalties'])
    if not stages:
        stages = ['preference']

    if time_limit is None:
        time_limit = float(meta['constraints'].get('max_time_seconds', 20))

    per_stage_limit = max(0.5, float(time_limit) / len(stages))

    solver = cp_model.CpSolver()
    stage_results: List[Dict[str, Any]] = []

    for stage_name in stages:
        if stage_name not in components:
            logger.warning('Stage %s not found in components, skipping', stage_name)
            continue

        expr = components[stage_name]
        direction = _stage_direction(stage_name)

        if direction == 'max':
            model.Maximize(expr)
        else:
            model.Minimize(expr)

        solver.parameters.max_time_in_seconds = per_stage_limit
        status = solver.Solve(model)

        if status not in FEASIBLE_STATUS:
            logger.error('Stage %s failed with status %s', stage_name, solver.StatusName(status))
            return None

        value = int(round(solver.Value(expr)))
        logger.debug('Stage %s (%s): value=%d', stage_name, direction, value)
        stage_results.append({
            'stage': stage_name,
            'direction': direction,
            'value': value,
            'status': status,
            'status_name': solver.StatusName(status),
        })

        model.Add(expr == value)

    if not stage_results:
        logger.error('No stages produced results')
        return None

    return build_assignment_dict(solver, x, meta, stage_results)
