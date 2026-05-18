from __future__ import annotations

from typing import Dict, Any, Tuple
from ortools.sat.python import cp_model


def build_model(data: Dict[str, Any]) -> Tuple[
    cp_model.CpModel,
    Dict[tuple[str, str], cp_model.IntVar],
    Dict[str, Any],
    Dict[str, Any],
]:
    people_data = data['people']
    team_sizes = data['team_sizes']
    weights = data['weights']
    constraints = data['constraints']

    people = list(people_data.keys())
    subsystems = list(team_sizes.keys())
    veterans = {name for name, info in people_data.items() if info.get('veteran', False)}
    scores = {name: int(info['score']) for name, info in people_data.items()}
    prefs = {name: dict(info['preferences']) for name, info in people_data.items()}

    model = cp_model.CpModel()
    x: Dict[tuple[str, str], cp_model.IntVar] = {}

    for p in people:
        for s in subsystems:
            x[(p, s)] = model.NewBoolVar(f'x_{p}_{s}')

    for p in people:
        model.AddExactlyOne(x[(p, s)] for s in subsystems)

    for s in subsystems:
        model.Add(sum(x[(p, s)] for p in people) == int(team_sizes[s]))

    min_veterans = int(constraints.get('min_veterans_per_team', 1))
    for s in subsystems:
        model.Add(sum(x[(p, s)] for p in veterans) >= min_veterans)

    pref_weights = weights['preference']
    penalty_4 = int(weights.get('penalties', {}).get('rank_4', 0))
    penalty_5 = int(weights.get('penalties', {}).get('rank_5', 0))
    high_perf_bonus = weights.get('high_performance_bonus', {})

    total_score = sum(scores.values())
    total_people = len(people)

    preference_expr = 0
    score_balance_expr = 0
    veteran_balance_expr = 0
    penalty_expr = 0

    for s in subsystems:
        team_score = sum(scores[p] * x[(p, s)] for p in people)
        team_size = int(team_sizes[s])

        score_dev = model.NewIntVar(0, total_score * total_people, f'score_dev_{s}')
        model.AddAbsEquality(score_dev, team_score * total_people - total_score * team_size)
        score_balance_expr += score_dev

        vet_count = sum(x[(p, s)] for p in veterans)
        veteran_dev = model.NewIntVar(0, len(veterans) * 10, f'veteran_dev_{s}')
        model.AddAbsEquality(veteran_dev, vet_count * 10 - min_veterans * 10)
        veteran_balance_expr += veteran_dev

    for p in people:
        for s in subsystems:
            rank = int(prefs[p][s])
            preference_expr += scores[p] * pref_weights[rank] * x[(p, s)]

            if rank == 4:
                penalty_expr += penalty_4 * x[(p, s)]
            elif rank == 5:
                penalty_expr += penalty_5 * x[(p, s)]

            if s in high_perf_bonus:
                preference_expr += int(high_perf_bonus[s] * scores[p]) * x[(p, s)]

    fairness_expr = score_balance_expr + veteran_balance_expr + penalty_expr

    objective_components = {
        'preference': preference_expr,
        'score_balance': score_balance_expr,
        'veteran_balance': veteran_balance_expr,
        'penalties': penalty_expr,
        'fairness': fairness_expr,
    }

    meta = {
        'people': people,
        'subsystems': subsystems,
        'scores': scores,
        'preferences': prefs,
        'veterans': veterans,
        'team_sizes': team_sizes,
        'weights': weights,
        'constraints': constraints,
        'optimization': data.get('optimization', {}),
        'pareto': data.get('pareto', {}),
        'metrics': data.get('metrics', {}),
    }

    return model, x, meta, objective_components
