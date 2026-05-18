from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    logger.info('Loading configuration from %s', path)
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f'Config file is empty: {path}')
    return normalize_config(data)


def normalize_config(data: Dict[str, Any]) -> Dict[str, Any]:
    people_raw = data.get('people', {})
    normalized_people: Dict[str, Dict[str, Any]] = {}

    for name, payload in people_raw.items():
        prefs = payload.get('preferences', {})
        prefs = {k.replace('TT_TC', 'TT&C'): int(v) for k, v in prefs.items()}

        normalized_people[name] = {
            'score': int(payload['score']),
            'veteran': bool(payload.get('veteran', False)),
            'preferences': prefs,
        }

    data['people'] = normalized_people

    if 'team_sizes' in data:
        data['team_sizes'] = {k.replace('TT_TC', 'TT&C'): int(v) for k, v in data['team_sizes'].items()}

    if 'weights' in data:
        weights = data['weights']
        if 'preference' in weights:
            weights['preference'] = {int(k): int(v) for k, v in weights['preference'].items()}
        if 'penalties' in weights:
            weights['penalties'] = {k: int(v) for k, v in weights['penalties'].items()}
        if 'objective' in weights:
            weights['objective'] = {k: int(v) for k, v in weights['objective'].items()}
        if 'high_performance_bonus' in weights:
            weights['high_performance_bonus'] = {k.replace('TT_TC', 'TT&C'): float(v) for k, v in weights['high_performance_bonus'].items()}

    if 'constraints' in data:
        cleaned_constraints = {}
        for k, v in data['constraints'].items():
            if isinstance(v, bool):
                cleaned_constraints[k] = v
            elif isinstance(v, int):
                cleaned_constraints[k] = v
            elif isinstance(v, float):
                cleaned_constraints[k] = v
            elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
                cleaned_constraints[k] = float(v) if '.' in v else int(v)
            else:
                cleaned_constraints[k] = v
        data['constraints'] = cleaned_constraints

    if 'optimization' in data:
        opt = data['optimization']
        if 'stages' in opt and isinstance(opt['stages'], list):
            opt['stages'] = [str(s) for s in opt['stages']]
        if 'mode' in opt:
            opt['mode'] = str(opt['mode'])

    if 'pareto' in data:
        pareto = data['pareto']
        if 'enabled' in pareto:
            pareto['enabled'] = bool(pareto['enabled'])
        if 'points' in pareto:
            pareto['points'] = int(pareto['points'])
        if 'min_preference_ratio' in pareto:
            pareto['min_preference_ratio'] = float(pareto['min_preference_ratio'])
        if 'time_limit_per_point' in pareto:
            pareto['time_limit_per_point'] = float(pareto['time_limit_per_point'])
        if 'fairness_weights' in pareto and isinstance(pareto['fairness_weights'], dict):
            pareto['fairness_weights'] = {k: int(v) for k, v in pareto['fairness_weights'].items()}

    if 'monte_carlo' in data:
        mc = data['monte_carlo']
        if 'enabled' in mc:
            mc['enabled'] = bool(mc['enabled'])
        if 'runs' in mc:
            mc['runs'] = int(mc['runs'])
        if 'score_jitter' in mc:
            mc['score_jitter'] = int(mc['score_jitter'])
        if 'preference_jitter' in mc:
            mc['preference_jitter'] = int(mc['preference_jitter'])
        if 'preference_noise_probability' in mc:
            mc['preference_noise_probability'] = float(mc['preference_noise_probability'])
        if 'time_limit_per_run' in mc:
            mc['time_limit_per_run'] = float(mc['time_limit_per_run'])
        if 'seed' in mc and mc['seed'] is not None:
            mc['seed'] = int(mc['seed'])

    logger.debug('Config normalized: %d people, %d teams',
                 len(data.get('people', {})), len(data.get('team_sizes', {})))
    return data
