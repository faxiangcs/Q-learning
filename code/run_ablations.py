"""Run the three C-to-F ablation studies and update the HTML report."""

import csv
import html
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path

from environment import Environment
from q_learning import TrainingSettings, train


ROOT = Path(__file__).resolve().parents[1]
ABLATION_CONFIG = ROOT / "config/ablations.json"
RESULTS_DIR = ROOT / "results/ablations"
ASSETS_DIR = ROOT / "report/assets"
REPORT_PATH = ROOT / "report/index.html"
METRICS = (
    "success_rate",
    "success_rate_last_500",
    "mean_return_last_500",
    "mean_steps_last_500",
    "final_q_max_abs_error",
    "final_q_rmse",
    "min_action_visits",
)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def optimal_q(environment: Environment, discount_factor: float) -> dict[tuple[str, str], float]:
    """Compute a separate dynamic-programming reference for this tiny deterministic graph."""
    values = {state: 0.0 for state in environment.states}
    for _ in range(10000):
        next_values = values.copy()
        for state in environment.states:
            actions = environment.adjacency[state]
            if state == environment.target_state:
                continue
            candidates = []
            for action in actions:
                if action == environment.target_state:
                    candidates.append(environment.target_reward)
                else:
                    candidates.append(environment.non_target_reward + discount_factor * values[action])
            next_values[state] = max(candidates)
        if max(abs(next_values[state] - values[state]) for state in values) < 1e-12:
            values = next_values
            break
        values = next_values
    reference = {}
    for state in environment.states:
        for action in environment.available_actions(state):
            reference[state, action] = (
                environment.target_reward if action == environment.target_state
                else environment.non_target_reward + discount_factor * values[action]
            )
    return reference


def summarize(result, reference: dict[tuple[str, str], float]) -> dict[str, float]:
    tail = result.records[-min(500, len(result.records)):]
    errors = [result.final_q[key] - reference[key] for key in reference]
    return {
        "episodes": len(result.records),
        "success_rate": statistics.fmean(record.reached_target for record in result.records),
        "success_rate_last_500": statistics.fmean(record.reached_target for record in tail),
        "mean_return_last_500": statistics.fmean(record.total_reward for record in tail),
        "mean_steps_last_500": statistics.fmean(record.steps for record in tail),
        "final_q_max_abs_error": max(abs(error) for error in errors),
        "final_q_rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
        "min_action_visits": min(result.visits.values()),
    }


def train_with_config(config: dict, seed: int, episodes: int, overrides: dict | None = None):
    experiment = deepcopy(config)
    experiment["training"]["random_seed"] = seed
    experiment["training"]["episodes"] = episodes
    if overrides:
        experiment["training"].update(overrides)
    environment = Environment.from_config(experiment)
    settings = TrainingSettings.from_config(experiment)
    result = train(environment, settings)
    reference = optimal_q(environment, settings.discount_factor)
    return environment, result, summarize(result, reference)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}"


def write_seed_study(config: dict, settings: dict):
    base = load_config(ROOT / settings["environment_config"])
    rows = []
    for seed in settings["seeds"]:
        environment, result, metrics = train_with_config(base, seed, base["training"]["episodes"])
        rows.append({"seed": seed, **metrics})
    fields = ["seed", *METRICS, "episodes"]
    write_csv(RESULTS_DIR / "random_seeds.csv", rows, fields)

    summary_rows = []
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        summary_rows.append({
            "metric": metric,
            "mean": statistics.fmean(values),
            "std_sample": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        })
    write_csv(RESULTS_DIR / "random_seed_summary.csv", summary_rows,
              ["metric", "mean", "std_sample", "min", "max"])
    return rows, summary_rows


def write_reference_q(config: dict) -> None:
    environment = Environment.from_config(config)
    reference = optimal_q(environment, float(config["training"]["discount_factor"]))
    rows = [
        {"state": state, "action_next_state": action, "q_star": value}
        for (state, action), value in reference.items()
    ]
    write_csv(RESULTS_DIR / "q_star_reference.csv", rows,
              ["state", "action_next_state", "q_star"])


def write_budget_study(config: dict, settings: dict):
    base = load_config(ROOT / settings["environment_config"])
    rows = []
    for episodes in settings["episodes"]:
        _, _, metrics = train_with_config(base, settings["seed"], episodes)
        rows.append({"episodes": episodes, "seed": settings["seed"], **metrics})
    write_csv(RESULTS_DIR / "training_budget.csv", rows,
              ["episodes", "seed", *METRICS])
    return rows


def write_exploration_study(config: dict, settings: dict):
    base = load_config(ROOT / settings["environment_config"])
    rows = []
    for strategy, values in settings["strategies"].items():
        _, _, metrics = train_with_config(
            base, settings["seed"], settings["episodes"], values
        )
        rows.append({
            "strategy": strategy,
            "label": values["label"],
            "epsilon_start": values["epsilon_start"],
            "epsilon_end": values["epsilon_end"],
            "epsilon_decay_episodes": values["epsilon_decay_episodes"],
            "seed": settings["seed"],
            "episodes": settings["episodes"],
            **metrics,
        })
    write_csv(RESULTS_DIR / "exploration_rates.csv", rows,
              ["strategy", "label", "epsilon_start", "epsilon_end",
               "epsilon_decay_episodes", "seed", "episodes", *METRICS])
    return rows


def svg_shell(title: str, width: int = 900, height: int = 520) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        '<desc id="desc">Results from a reproducible Q-learning ablation study. The numerical source is saved in the results/ablations directory.</desc>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#263238">',
        f'<text x="75" y="38" font-size="22" font-weight="bold">{html.escape(title)}</text>',
    ]


def add_axes(parts: list[str], left: float, right: float, top: float, bottom: float,
             y_min: float, y_max: float, log_scale: bool = False) -> None:
    def to_y(value: float) -> float:
        if log_scale:
            value = math.log10(max(value, 1e-12))
            y_low, y_high = math.log10(max(y_min, 1e-12)), math.log10(max(y_max, 1e-12))
        else:
            y_low, y_high = y_min, y_max
        return bottom - (value - y_low) / (y_high - y_low) * (bottom - top)
    for tick in range(5):
        if log_scale:
            low, high = math.log10(max(y_min, 1e-12)), math.log10(max(y_max, 1e-12))
            log_value = low + (high - low) * tick / 4
            value = 10 ** log_value
        else:
            value = y_min + (y_max - y_min) * tick / 4
        y = to_y(value)
        label = f"{value:.2g}"
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e2e8e8"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{label}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#657275"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#657275"/>')


def plot_bars(path: Path, title: str, labels: list[str], values: list[float],
              y_label: str, log_scale: bool = False, mean_value: float | None = None) -> None:
    width, height = 900, 520
    left, right, top, bottom = 100, 850, 90, 430
    parts = svg_shell(title, width, height)
    minimum = max(min(values) * 0.7, 1e-12) if log_scale else min(0.0, min(values))
    maximum = max(values) * 1.18 if max(values) else 1.0
    if not log_scale and maximum == minimum:
        maximum = minimum + 1
    add_axes(parts, left, right, top, bottom, minimum, maximum, log_scale)
    def to_y(value: float) -> float:
        if log_scale:
            low, high = math.log10(max(minimum, 1e-12)), math.log10(max(maximum, 1e-12))
            current = math.log10(max(value, 1e-12))
        else:
            low, high, current = minimum, maximum, value
        return bottom - (current - low) / (high - low) * (bottom - top)
    span = (right - left) / max(1, len(values))
    bar_width = span * 0.56
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + span * index + (span - bar_width) / 2
        y = to_y(value)
        base = bottom if not log_scale else to_y(minimum)
        parts.append(f'<rect x="{x:.2f}" y="{min(y, base):.2f}" width="{bar_width:.2f}" height="{abs(base-y):.2f}" fill="#187a72"/>')
        parts.append(f'<text x="{x + bar_width/2:.2f}" y="{bottom + 24}" font-size="12" text-anchor="middle">{html.escape(label)}</text>')
        parts.append(f'<text x="{x + bar_width/2:.2f}" y="{max(top - 8, y - 8):.2f}" font-size="12" text-anchor="middle">{fmt(value)}</text>')
    if mean_value is not None:
        y = to_y(mean_value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#b8543c" stroke-width="2" stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{right}" y="{y - 8:.2f}" font-size="12" text-anchor="end">mean {fmt(mean_value)}</text>')
    parts.append(f'<text x="{(left+right)/2:.0f}" y="485" font-size="14" text-anchor="middle">Study condition</text>')
    parts.append(f'<text x="22" y="{(top+bottom)/2:.0f}" font-size="14" text-anchor="middle" transform="rotate(-90 22 {(top+bottom)/2:.0f})">{html.escape(y_label)}</text>')
    parts.extend(('</g></svg>',))
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def plot_budget(path: Path, rows: list[dict]) -> None:
    labels = [str(row["episodes"]) for row in rows]
    values = [float(row["final_q_max_abs_error"]) for row in rows]
    plot_bars(path, "Training budget: final Q error (log scale)", labels, values,
              "max |Q - Q*|", log_scale=True)


def plot_exploration(path: Path, rows: list[dict]) -> None:
    labels = [row["strategy"] for row in rows]
    values = [float(row["final_q_max_abs_error"]) for row in rows]
    plot_bars(path, "Exploration schedules: final Q error (log scale)", labels, values,
              "max |Q - Q*|", log_scale=True)


def update_report(seed_rows: list[dict], seed_summary: list[dict], budget_rows: list[dict], exploration_rows: list[dict]) -> None:
    document = REPORT_PATH.read_text(encoding="utf-8")
    start_marker = "<!-- ABLATION_RESULTS_START -->"
    end_marker = "<!-- ABLATION_RESULTS_END -->"
    if document.count(start_marker) != 1 or document.count(end_marker) != 1:
        raise ValueError("report is missing unique ablation markers")

    def table(headers: list[str], rows: list[list[str]]) -> str:
        content = ['    <div class="table-wrap"><table>', '      <thead><tr>']
        content.extend(f'<th scope="col">{html.escape(header)}</th>' for header in headers)
        content.extend(('</tr></thead>', '      <tbody>'))
        for row in rows:
            content.append('        <tr>' + ''.join(f'<td>{html.escape(str(cell))}</td>' for cell in row) + '</tr>')
        content.extend(('      </tbody>', '    </table></div>'))
        return "\n".join(content)

    seed_headers = ["Seed", "Success rate", "Last-500 return", "Last-500 steps", "Max Q error", "Min visits"]
    seed_table = table(seed_headers, [
        [row["seed"], fmt(row["success_rate"]), fmt(row["mean_return_last_500"]),
         fmt(row["mean_steps_last_500"]), fmt(row["final_q_max_abs_error"], 4), row["min_action_visits"]]
        for row in seed_rows
    ])
    summary_by_metric = {row["metric"]: row for row in seed_summary}
    summary_table = table(["Metric", "Mean", "Sample standard deviation", "Min", "Max"], [
        [metric, fmt(summary_by_metric[metric]["mean"]), fmt(summary_by_metric[metric]["std_sample"]),
         fmt(summary_by_metric[metric]["min"]), fmt(summary_by_metric[metric]["max"])]
        for metric in ("success_rate", "mean_return_last_500", "mean_steps_last_500", "final_q_max_abs_error", "min_action_visits")
    ])
    budget_table = table(["Episodes", "Success rate", "Last-500 return", "Last-500 steps", "Max Q error", "Min visits"], [
        [row["episodes"], fmt(row["success_rate"]), fmt(row["mean_return_last_500"]),
         fmt(row["mean_steps_last_500"]), fmt(row["final_q_max_abs_error"], 4), row["min_action_visits"]]
        for row in budget_rows
    ])
    exploration_table = table(["Strategy", "ε settings", "Success rate", "Last-500 steps", "Max Q error", "Min visits"], [
        [row["label"], f'{row["epsilon_start"]} → {row["epsilon_end"]} / {row["epsilon_decay_episodes"]}',
         fmt(row["success_rate"]), fmt(row["mean_steps_last_500"]), fmt(row["final_q_max_abs_error"], 4), row["min_action_visits"]]
        for row in exploration_rows
    ])

    seed_error = summary_by_metric["final_q_max_abs_error"]
    budget_start, budget_end = budget_rows[0], budget_rows[-1]
    current = next(row for row in exploration_rows if row["strategy"] == "current_decay")
    fixed = next(row for row in exploration_rows if row["strategy"] == "fixed_0.1")
    fast = next(row for row in exploration_rows if row["strategy"] == "fast_decay")
    sections = [
        '    <h3>1. Random-seed repetition: C to F</h3>',
        '    <p>Five independent runs use seeds 0, 1, 2, 3, and 42. All hyperparameters and the 10,000-episode budget remain fixed. Success rate is the fraction of episodes reaching F. The Q error is the maximum absolute difference between the learned final table and an independently computed dynamic-programming reference Q*. It is a diagnostic, not an input to training.</p>',
        seed_table,
        '    <p>Across the five seeds, the mean final max Q error was ' + fmt(seed_error["mean"], 4) + ' with sample standard deviation ' + fmt(seed_error["std_sample"], 4) + '. The standard deviation and the minimum action-visit count show that seed choice affects coverage of less frequently reached regions, even though the success rate is saturated.</p>',
        summary_table,
        '    <figure><img src="assets/ablation_random_seeds.svg" alt="Bar chart of final Q error for five random seeds"><figcaption>Each bar is one independently seeded C-to-F run; the dashed line is the five-seed mean. The exact runs and summary statistics are in <code>results/ablations/random_seeds.csv</code> and <code>results/ablations/random_seed_summary.csv</code>.</figcaption></figure>',
        '    <h3>2. Training budget: seed 42</h3>',
        '    <p>Keeping seed 42 and all other settings fixed, the budget varies from 500 to 10,000 episodes. This separates early apparent success from learning of the full table. The error falls from ' + fmt(budget_start["final_q_max_abs_error"], 4) + ' at 500 episodes to ' + fmt(budget_end["final_q_max_abs_error"], 4) + ' at 10,000 episodes, while success is already 100% at 500 episodes. The success metric therefore cannot by itself establish that the full table has been learned.</p>',
        budget_table,
        '    <figure><img src="assets/ablation_training_budget.svg" alt="Bar chart of final Q error for five training budgets"><figcaption>The budget plot uses a logarithmic vertical axis because the error decreases by orders of magnitude. The exact values are in <code>results/ablations/training_budget.csv</code>.</figcaption></figure>',
        '    <h3>3. Exploration-rate ablation: seed 42</h3>',
        '    <p>The current schedule is compared with fixed ε = 0.1 and a fast decay from 0.4 to 0.01 over 2,000 episodes. All three runs use 10,000 episodes. In this run, the current schedule has max Q error ' + fmt(current["final_q_max_abs_error"], 4) + ' with a minimum of ' + str(current["min_action_visits"]) + ' visits per action; fixed ε has error ' + fmt(fixed["final_q_max_abs_error"], 4) + ' and only ' + str(fixed["min_action_visits"]) + ' visits for its least-sampled action; fast decay has error ' + fmt(fast["final_q_max_abs_error"], 4) + ' and ' + str(fast["min_action_visits"]) + ' minimum visits. All three still reach F in every episode. This directly shows why apparent task success can be confused with complete state-action coverage.</p>',
        exploration_table,
        '    <figure><img src="assets/ablation_exploration.svg" alt="Bar chart of final Q error for three exploration schedules"><figcaption>The exploration plot uses a logarithmic vertical axis. The exact schedules and metrics are in <code>results/ablations/exploration_rates.csv</code>.</figcaption></figure>',
        '    <h3>Interpretation and limitations</h3>',
        '    <p>These ablations separate three sources of uncertainty: random initialization of the sampling trajectory, insufficient training budget, and the exploration schedule. The comparison is limited by the small deterministic graph, the manually interpreted adjacency, the fixed reward scale, and the use of one target and one start state in the ablations. Q* is available here only because this six-state deterministic environment is simple enough for an independent dynamic-programming check; in a larger unknown environment, such a reference may not be available. Observed stabilization and high return therefore remain empirical evidence rather than a guarantee of optimality.</p>',
    ]
    before, rest = document.split(start_marker, 1)
    _, after = rest.split(end_marker, 1)
    REPORT_PATH.write_text(
        before + start_marker + "\n" + "\n".join(sections) + "\n    " + end_marker + after,
        encoding="utf-8",
    )


def main() -> None:
    config = load_config(ABLATION_CONFIG)
    baseline_config = load_config(ROOT / config["random_seed_repetition"]["environment_config"])
    write_reference_q(baseline_config)
    seed_rows, seed_summary = write_seed_study(config, config["random_seed_repetition"])
    budget_rows = write_budget_study(config, config["training_budget"])
    exploration_rows = write_exploration_study(config, config["exploration_rate"])
    plot_bars(
        ASSETS_DIR / "ablation_random_seeds.svg",
        "Random-seed repetition: final Q error",
        [str(row["seed"]) for row in seed_rows],
        [float(row["final_q_max_abs_error"]) for row in seed_rows],
        "max |Q - Q*|",
        log_scale=True,
        mean_value=statistics.fmean(float(row["final_q_max_abs_error"]) for row in seed_rows),
    )
    plot_budget(ASSETS_DIR / "ablation_training_budget.svg", budget_rows)
    plot_exploration(ASSETS_DIR / "ablation_exploration.svg", exploration_rows)
    update_report(seed_rows, seed_summary, budget_rows, exploration_rows)
    print("Random-seed repetition:")
    for row in seed_rows:
        print(row)
    print("Training budget:")
    for row in budget_rows:
        print(row)
    print("Exploration rate:")
    for row in exploration_rows:
        print(row)
    print(f"Results: {RESULTS_DIR}")
    print(f"Plots: {ASSETS_DIR}/ablation_*.svg")


if __name__ == "__main__":
    main()
