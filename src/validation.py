from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

KNOWN_MODES = {'lexicographic', 'pareto', 'weighted'}
KNOWN_STAGES = {'preference', 'score_balance', 'veteran_balance', 'penalties'}


@dataclass
class ConfigValidationError(Exception):
    errors: List[str]

    def __str__(self) -> str:
        return '\n'.join(self.errors)


def _as_int(value: Any, field: str, errors: List[str]) -> int | None:
    try:
        return int(value)
    except Exception:
        errors.append(f'{field} must be an integer, got {value!r}.')
        return None


def _as_float(value: Any, field: str, errors: List[str]) -> float | None:
    try:
        return float(value)
    except Exception:
        errors.append(f'{field} must be numeric, got {value!r}.')
        return None


def validate_config(data: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []

    if not isinstance(data, dict):
        raise ConfigValidationError(['Configuration root must be a mapping/dictionary.'])

    people = data.get('people')
    team_sizes = data.get('team_sizes')
    weights = data.get('weights', {})
    constraints = data.get('constraints', {})
    optimization = data.get('optimization', {})
    pareto = data.get('pareto', {})
    monte_carlo = data.get('monte_carlo', {})

    if not isinstance(people, dict) or not people:
        errors.append('"people" must be a non-empty mapping.')
    if not isinstance(team_sizes, dict) or not team_sizes:
        errors.append('"team_sizes" must be a non-empty mapping.')

    if errors:
        raise ConfigValidationError(errors)

    subsystem_names = list(team_sizes.keys())

    total_team_size = 0
    for team, size in team_sizes.items():
        size_i = _as_int(size, f'team_sizes.{team}', errors)
        if size_i is not None and size_i <= 0:
            errors.append(f'team_sizes.{team} must be > 0, got {size_i}.')
        if size_i is not None:
            total_team_size += size_i

    veteran_count = 0
    for name, payload in people.items():
        if not isinstance(payload, dict):
            errors.append(f'people.{name} must be a mapping.')
            continue

        score = _as_int(payload.get('score'), f'people.{name}.score', errors)
        if score is not None and score < 0:
            errors.append(f'people.{name}.score must be >= 0, got {score}.')

        veteran = payload.get('veteran', False)
        if not isinstance(veteran, bool):
            errors.append(f'people.{name}.veteran must be boolean, got {veteran!r}.')
        elif veteran:
            veteran_count += 1

        prefs = payload.get('preferences')
        if not isinstance(prefs, dict):
            errors.append(f'people.{name}.preferences must be a mapping.')
        else:
            missing = set(subsystem_names) - set(prefs.keys())
            extra = set(prefs.keys()) - set(subsystem_names)
            if missing:
                errors.append(f'people.{name}.preferences missing subsystems: {sorted(missing)}.')
            if extra:
                errors.append(f'people.{name}.preferences contains unknown subsystems: {sorted(extra)}.')

            for subsystem in subsystem_names:
                if subsystem not in prefs:
                    continue
                rank = _as_int(prefs[subsystem], f'people.{name}.preferences.{subsystem}', errors)
                if rank is not None and rank not in {1, 2, 3, 4, 5}:
                    errors.append(f'people.{name}.preferences.{subsystem} must be in 1..5, got {rank}.')

    if 'constraints' in data:
        min_vets = constraints.get('min_veterans_per_team', 1)
        min_vets_i = _as_int(min_vets, 'constraints.min_veterans_per_team', errors)
        if min_vets_i is not None and min_vets_i < 0:
            errors.append('constraints.min_veterans_per_team must be >= 0.')
        if min_vets_i is not None and len(team_sizes) * min_vets_i > veteran_count:
            errors.append(
                f'Not enough veterans to satisfy the minimum per team: '
                f'{veteran_count} veterans for {len(team_sizes)} teams with minimum {min_vets_i} each.'
            )

        max_time = constraints.get('max_time_seconds', 20)
        max_time_f = _as_float(max_time, 'constraints.max_time_seconds', errors)
        if max_time_f is not None and max_time_f <= 0:
            errors.append('constraints.max_time_seconds must be a positive number.')

    if 'optimization' in data:
        mode = str(optimization.get('mode', 'lexicographic')).lower()
        if mode not in KNOWN_MODES:
            errors.append(f'optimization.mode must be one of {sorted(KNOWN_MODES)}, got {mode!r}.')

        stages = optimization.get('stages', ['preference', 'score_balance', 'veteran_balance', 'penalties'])
        if not isinstance(stages, list) or not stages:
            errors.append('optimization.stages must be a non-empty list.')
        else:
            invalid = [s for s in stages if s not in KNOWN_STAGES]
            if invalid:
                errors.append(f'optimization.stages contains invalid entries: {invalid}.')

    if 'weights' in data:
        pref_weights = weights.get('preference', {})
        if not isinstance(pref_weights, dict) or set(map(int, pref_weights.keys())) != {1, 2, 3, 4, 5}:
            errors.append('weights.preference must define integer weights for ranks 1..5.')

        penalties = weights.get('penalties', {})
        if penalties and not isinstance(penalties, dict):
            errors.append('weights.penalties must be a mapping if provided.')

        hpb = weights.get('high_performance_bonus', {})
        if hpb and not isinstance(hpb, dict):
            errors.append('weights.high_performance_bonus must be a mapping if provided.')

    if pareto:
        if not isinstance(pareto, dict):
            errors.append('pareto must be a mapping if provided.')
        else:
            enabled = pareto.get('enabled', False)
            if not isinstance(enabled, bool):
                errors.append('pareto.enabled must be boolean.')

            if enabled or str(optimization.get('mode', 'lexicographic')).lower() == 'pareto':
                points = pareto.get('points', 6)
                points_i = _as_int(points, 'pareto.points', errors)
                if points_i is not None and points_i < 2:
                    errors.append('pareto.points must be >= 2 when Pareto frontier is enabled.')

                ratio = pareto.get('min_preference_ratio', 0.9)
                ratio_f = _as_float(ratio, 'pareto.min_preference_ratio', errors)
                if ratio_f is not None and not (0 < ratio_f <= 1):
                    errors.append('pareto.min_preference_ratio must be in (0, 1].')

                ppt = pareto.get('time_limit_per_point', 8)
                ppt_f = _as_float(ppt, 'pareto.time_limit_per_point', errors)
                if ppt_f is not None and ppt_f <= 0:
                    errors.append('pareto.time_limit_per_point must be positive.')

                fw = pareto.get('fairness_weights', {})
                if fw and not isinstance(fw, dict):
                    errors.append('pareto.fairness_weights must be a mapping if provided.')

    if monte_carlo:
        if not isinstance(monte_carlo, dict):
            errors.append('monte_carlo must be a mapping if provided.')
        else:
            enabled = monte_carlo.get('enabled', False)
            if not isinstance(enabled, bool):
                errors.append('monte_carlo.enabled must be boolean.')

            runs = monte_carlo.get('runs', 50)
            runs_i = _as_int(runs, 'monte_carlo.runs', errors)
            if runs_i is not None and runs_i < 1:
                errors.append('monte_carlo.runs must be >= 1.')

            jitter = monte_carlo.get('score_jitter', 4)
            jitter_i = _as_int(jitter, 'monte_carlo.score_jitter', errors)
            if jitter_i is not None and jitter_i < 0:
                errors.append('monte_carlo.score_jitter must be >= 0.')

            pj = monte_carlo.get('preference_jitter', 1)
            pj_i = _as_int(pj, 'monte_carlo.preference_jitter', errors)
            if pj_i is not None and pj_i < 0:
                errors.append('monte_carlo.preference_jitter must be >= 0.')

            prob = monte_carlo.get('preference_noise_probability', 0.05)
            prob_f = _as_float(prob, 'monte_carlo.preference_noise_probability', errors)
            if prob_f is not None and not (0.0 <= prob_f <= 1.0):
                errors.append('monte_carlo.preference_noise_probability must be in [0, 1].')

            tl = monte_carlo.get('time_limit_per_run', 5)
            tl_f = _as_float(tl, 'monte_carlo.time_limit_per_run', errors)
            if tl_f is not None and tl_f <= 0:
                errors.append('monte_carlo.time_limit_per_run must be positive.')

            seed = monte_carlo.get('seed', None)
            if seed is not None:
                _as_int(seed, 'monte_carlo.seed', errors)

    if total_team_size != len(people):
        errors.append(
            f'Sum of team sizes ({total_team_size}) must equal the number of people ({len(people)}).')

    if errors:
        logger.error('Config validation failed with %d errors', len(errors))
        raise ConfigValidationError(errors)

    logger.info('Config validation passed')
    return data
