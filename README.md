# CP-SAT Team Allocation

> Constraint-based team allocation solver using Google OR-Tools CP-SAT, with lexicographic optimization, Pareto frontier analysis, and Monte Carlo robustness testing.

Built for allocating people to subsystem teams while balancing preferences, technical scores, veteran distribution, and fairness.

## How It Works

1. **Model** - builds a CP-SAT model from a YAML config: binary variables for each (person, subsystem) pair, with hard constraints for team sizes and minimum veteran counts
2. **Lexicographic optimization** - solves objectives one at a time, freezing each optimum before moving to the next: `preference -> score_balance -> veteran_balance -> penalties`
3. **Pareto frontier** (optional) - samples trade-off solutions between preference satisfaction and fairness metrics. Identifies the knee point (best trade-off) automatically
4. **Monte Carlo** (optional) - runs the solver many times under random input perturbations to measure assignment stability
5. **Knee point** - the Pareto point with the best normalized trade-off between preference and fairness, highlighted in both the report and the scatter plot
6. **Representative distribution** - mode assignment across Monte Carlo runs, showing where each person most often ends up

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## CLI Options

### Basic
| Flag | Description |
|---|---|
| `--config PATH` | Config file (default: config.yaml) |
| `--output PATH` | Report output (default: allocation_report.txt) |
| `--quiet` | Suppress progress bars and info messages |
| `--verbose` | Show debug-level logs |

### Pareto Frontier
| Flag | Description |
|---|---|
| `--pareto` | Enable Pareto frontier generation |
| `--pareto-points N` | Number of frontier points |
| `--pareto-min-ratio F` | Minimum preference ratio (0-1) |
| `--pareto-time-limit F` | Time limit per point (seconds) |

### Monte Carlo
| Flag | Description |
|---|---|
| `--monte-carlo` | Enable Monte Carlo analysis |
| `--monte-carlo-runs N` | Number of trials |
| `--monte-carlo-seed N` | Random seed |
| `--monte-carlo-time-limit F` | Time limit per run (seconds) |
| `--mc-score-jitter N` | Score perturbation range |
| `--mc-preference-jitter N` | Preference rank perturbation range |
| `--mc-noise-prob F` | Preference noise probability (0-1) |

### Solver
| Flag | Description |
|---|---|
| `--max-time F` | Total solver time limit (seconds) |
| `--min-veterans N` | Minimum veterans per team |

### Metrics
| Flag | Description |
|---|---|
| `--exclude-veterans-avg-score` | Exclude veterans from team average score calculation |

### Examples

```bash
# Basic usage
python main.py

# With Pareto frontier and Monte Carlo
python main.py --pareto --monte-carlo --monte-carlo-runs 100

# Custom config and output
python main.py --config custom.yaml --output result.txt

# Exclude veterans from average score calculation
python main.py --exclude-veterans-avg-score

# Full analysis with custom parameters
python main.py --pareto --pareto-points 12 --pareto-min-ratio 0.85 \
               --monte-carlo --monte-carlo-runs 200 --mc-noise-prob 0.1 \
               --max-time 60 --verbose

# Quiet mode (progress bars hidden)
python main.py --pareto --monte-carlo --quiet
```

## Output

- **allocation_report.txt** - full report with team distributions, fairness metrics, Pareto table, knee point distribution, and Monte Carlo stats
- **pareto_frontier.png** - scatter plot of trade-off solutions, knee point highlighted in red
- **team_composition_heatmap.png** - per-team metrics normalized heatmap
- **stability_heatmap.png** - per-person assignment probability across MC runs
- **assignment_confidence.png** - baseline retention barplot

## Project Structure

```
cpsat_allocator/
├── main.py               # Entry point and CLI
├── config.yaml           # Default problem definition
├── custom.yaml           # Alternative configuration
├── requirements.txt
└── src/
    ├── model.py          # CP-SAT model construction
    ├── solver.py         # Lexicographic solver
    ├── solution.py       # Shared solution helpers
    ├── pareto.py         # Pareto frontier + knee point
    ├── robustness.py     # Monte Carlo analysis (parallel)
    ├── data_loader.py    # YAML config loading
    ├── validation.py     # Config validation
    ├── metrics.py        # Statistics and fairness
    ├── report.py         # Report generation
    └── visualization.py  # Chart generation
```

## Tech Stack

- **Solver**: Google OR-Tools CP-SAT
- **Config**: PyYAML
- **Charts**: Matplotlib
- **Language**: Python 3.10+

## License

Licensed under GPL v3.0 - see [`LICENSE`](./LICENSE) for details.
