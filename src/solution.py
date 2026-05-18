from __future__ import annotations

from typing import Any, Dict, List
from ortools.sat.python import cp_model


FEASIBLE_STATUS = {cp_model.OPTIMAL, cp_model.FEASIBLE}


def build_assignment_dict(
    solver: cp_model.CpSolver,
    x: Dict[tuple[str, str], cp_model.IntVar],
    meta: Dict[str, Any],
    stage_results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    assignment: Dict[str, str] = {}
    team_members = {s: [] for s in meta['subsystems']}

    for p in meta['people']:
        for s in meta['subsystems']:
            if solver.Value(x[(p, s)]) == 1:
                assignment[p] = s
                team_members[s].append(p)
                break

    return {
        'status': stage_results[-1]['status'] if stage_results else None,
        'status_name': stage_results[-1]['status_name'] if stage_results else 'UNKNOWN',
        'objective': stage_results[0]['value'] if stage_results else 0,
        'stage_results': stage_results or [],
        'assignment': assignment,
        'team_members': team_members,
        'solver': solver,
        'x': x,
        'meta': meta,
    }
