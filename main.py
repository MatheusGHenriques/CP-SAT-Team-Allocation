from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.data_loader import load_config
from src.pareto import generate_pareto_frontier, find_knee_point
from src.report import generate_report
from src.robustness import run_monte_carlo_robustness
from src.solver import solve_lexicographic
from src.metrics import compute_team_stats
from src.validation import ConfigValidationError, validate_config
from src.visualization import (
    generate_assignment_confidence_barplot,
    generate_assignment_stability_heatmap,
    generate_pareto_frontier_scatter,
    generate_team_composition_heatmap,
)


def _apply_cli_overrides(data: dict, args: argparse.Namespace) -> None:
    if args.max_time is not None:
        data.setdefault('constraints', {})['max_time_seconds'] = args.max_time
    if args.min_veterans is not None:
        data.setdefault('constraints', {})['min_veterans_per_team'] = args.min_veterans
    if args.pareto_min_ratio is not None:
        data.setdefault('pareto', {})['min_preference_ratio'] = args.pareto_min_ratio
    if args.pareto_time_limit is not None:
        data.setdefault('pareto', {})['time_limit_per_point'] = args.pareto_time_limit
    if args.mc_score_jitter is not None:
        data.setdefault('monte_carlo', {})['score_jitter'] = args.mc_score_jitter
    if args.mc_preference_jitter is not None:
        data.setdefault('monte_carlo', {})['preference_jitter'] = args.mc_preference_jitter
    if args.mc_noise_prob is not None:
        data.setdefault('monte_carlo', {})['preference_noise_probability'] = args.mc_noise_prob
    if args.exclude_veterans_avg_score is not None:
        data.setdefault('metrics', {})['exclude_veterans_from_avg_score'] = args.exclude_veterans_avg_score


def _setup_logging(quiet: bool, verbose: bool) -> None:
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(
        description='CP-SAT team allocation solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python main.py\n'
            '  python main.py --pareto --pareto-points 8\n'
            '  python main.py --monte-carlo --monte-carlo-runs 100\n'
            '  python main.py --config custom.yaml --output report.txt --verbose\n'
            '  python main.py --pareto --mc-noise-prob 0.1 --max-time 60\n'
        ),
    )

    cfg = parser.add_argument_group('Configuration')
    cfg.add_argument('--config', default='config.yaml',
                     help='Path to YAML config file (default: config.yaml)')
    cfg.add_argument('--output', default='allocation_report.txt',
                     help='Report output file (default: allocation_report.txt)')

    pareto_grp = parser.add_argument_group('Pareto frontier')
    pareto_grp.add_argument('--pareto', action='store_true',
                            help='Enable Pareto frontier generation')
    pareto_grp.add_argument('--pareto-points', type=int, default=None,
                            help='Number of frontier points (default: from config)')
    pareto_grp.add_argument('--pareto-min-ratio', type=float, default=None,
                            help='Minimum preference ratio for Pareto range (0-1)')
    pareto_grp.add_argument('--pareto-time-limit', type=float, default=None,
                            help='Solver time limit per Pareto point (seconds)')

    mc_grp = parser.add_argument_group('Monte Carlo')
    mc_grp.add_argument('--monte-carlo', action='store_true',
                        help='Enable Monte Carlo robustness analysis')
    mc_grp.add_argument('--monte-carlo-runs', type=int, default=None,
                        help='Number of Monte Carlo trials')
    mc_grp.add_argument('--monte-carlo-seed', type=int, default=None,
                        help='Random seed for reproducibility')
    mc_grp.add_argument('--monte-carlo-time-limit', type=float, default=None,
                        help='Solver time limit per MC run (seconds)')
    mc_grp.add_argument('--mc-score-jitter', type=int, default=None,
                        help='Score perturbation range (±)')
    mc_grp.add_argument('--mc-preference-jitter', type=int, default=None,
                        help='Preference rank perturbation range (±)')
    mc_grp.add_argument('--mc-noise-prob', type=float, default=None,
                        help='Probability of perturbing a preference (0-1)')

    solver_grp = parser.add_argument_group('Solver')
    solver_grp.add_argument('--max-time', type=float, default=None,
                            help='Total solver time limit (seconds)')
    solver_grp.add_argument('--min-veterans', type=int, default=None,
                            help='Minimum veterans per team')

    output_grp = parser.add_argument_group('Output')
    output_grp.add_argument('--quiet', action='store_true',
                            help='Suppress progress bars and info messages')
    output_grp.add_argument('--verbose', action='store_true',
                            help='Show debug-level log messages')

    metrics_grp = parser.add_argument_group('Metrics')
    metrics_grp.add_argument('--exclude-veterans-avg-score', action='store_true', default=None,
                             help='Exclude veterans from team average score calculation')

    args = parser.parse_args()
    _setup_logging(args.quiet, args.verbose)
    logger = logging.getLogger('cpsat')

    config_path = Path(args.config)
    logger.info('Loading config from %s', config_path)
    data = load_config(config_path)

    _apply_cli_overrides(data, args)

    try:
        validate_config(data)
    except ConfigValidationError as exc:
        logger.error('Configuration validation failed')
        print('Invalid configuration:', file=sys.stderr)
        for err in exc.errors:
            print(f'  - {err}', file=sys.stderr)
        raise SystemExit(1)

    logger.info('Solving allocation (lexicographic optimization)')
    solution = solve_lexicographic(data)
    if solution is None:
        logger.error('No feasible solution found')
        print('No feasible solution found. Check constraints.', file=sys.stderr)
        raise SystemExit(1)
    logger.info('Solution found with objective=%s', solution['objective'])

    pareto_cfg = data.get('pareto', {}) or {}
    frontier_enabled = (
        args.pareto
        or bool(pareto_cfg.get('enabled', False))
        or str(data.get('optimization', {}).get('mode', 'lexicographic')).lower()
        == 'pareto'
    )
    if frontier_enabled:
        logger.info('Generating Pareto frontier')
        frontier = generate_pareto_frontier(
            data,
            max_points=args.pareto_points or pareto_cfg.get('points', 6),
            min_preference_ratio=pareto_cfg.get('min_preference_ratio', 0.9),
            time_limit_per_point=pareto_cfg.get('time_limit_per_point', 8),
            quiet=args.quiet,
        )
        solution['pareto_frontier'] = frontier
        solution['pareto_knee'] = find_knee_point(frontier) if frontier else None
        if frontier:
            logger.info('Pareto frontier: %d points, knee point found', len(frontier))

    mc_cfg = data.get('monte_carlo', {}) or {}
    monte_enabled = args.monte_carlo or bool(mc_cfg.get('enabled', False))
    if monte_enabled:
        logger.info('Running Monte Carlo robustness analysis')
        solution['monte_carlo'] = run_monte_carlo_robustness(
            data,
            baseline_solution=solution,
            runs=args.monte_carlo_runs if args.monte_carlo_runs is not None else mc_cfg.get('runs', 50),
            score_jitter=mc_cfg.get('score_jitter', 4),
            preference_jitter=mc_cfg.get('preference_jitter', 1),
            preference_noise_probability=mc_cfg.get('preference_noise_probability', 0.05),
            time_limit_per_run=args.monte_carlo_time_limit if args.monte_carlo_time_limit is not None else mc_cfg.get('time_limit_per_run', 5),
            seed=args.monte_carlo_seed if args.monte_carlo_seed is not None else mc_cfg.get('seed', None),
            quiet=args.quiet,
        )
        mc = solution['monte_carlo']
        logger.info('Monte Carlo: %d/%d successful',
                    mc.get('successful_runs', 0), mc.get('failed_runs', 0) + mc.get('successful_runs', 0))

    logger.info('Generating report')
    report = generate_report(solution)
    print(report)

    report_path = Path(args.output)
    report_path.write_text(report, encoding='utf-8')
    logger.info('Report saved to %s', report_path)

    solution['team_stats'] = compute_team_stats(solution)

    if 'pareto_frontier' in solution and solution['pareto_frontier']:
        logger.debug('Generating Pareto scatter plot')
        generate_pareto_frontier_scatter(
            solution['pareto_frontier'],
            knee_point=solution.get('pareto_knee'),
        )

    logger.debug('Generating team composition heatmap')
    generate_team_composition_heatmap(solution['team_stats'])

    if 'monte_carlo' in solution and solution['monte_carlo']:
        logger.debug('Generating Monte Carlo visualizations')
        generate_assignment_stability_heatmap(
            solution['monte_carlo'],
            list(data['people'].keys()),
            list(data['team_sizes'].keys()),
        )
        generate_assignment_confidence_barplot(solution['monte_carlo'])

    logger.info('Done')


if __name__ == '__main__':
    main()
