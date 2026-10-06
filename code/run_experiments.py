"""Run configured experiments and export reproducible tables and SVG plots."""

import argparse
import csv
import html
import json
from pathlib import Path

from environment import Environment
from q_learning import TrainingSettings, train


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    ROOT / "config/baseline.json",
    ROOT / "config/start_A_goal_C.json",
)
PLOT_WIDTH = 900
PLOT_HEIGHT = 600


def write_q_table(path: Path, environment: Environment, q: dict, visits: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("state", "action_next_state", "q_value", "visit_count"))
        for state in environment.states:
            for action in environment.available_actions(state):
                writer.writerow((state, action, f"{q[state, action]:.12g}", visits[state, action]))


def write_convergence(path: Path, environment: Environment, records: list) -> None:
    keys = [
        (state, action)
        for state in environment.states
        for action in environment.available_actions(state)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow((
            "episode", "epsilon", "total_reward", "steps", "reached_target",
            "max_abs_q_change", *(f"Q_{state}_{action}" for state, action in keys)
        ))
        for record in records:
            writer.writerow((
                record.episode, f"{record.epsilon:.12g}",
                f"{record.total_reward:.12g}", record.steps,
                int(record.reached_target), f"{record.max_abs_q_change:.12g}",
                *(f"{record.q_values[key]:.12g}" for key in keys)
            ))


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def draw_plot(path: Path, environment: Environment, result) -> None:
    records = result.records
    actions = environment.available_actions(environment.start_state)
    all_keys = tuple(result.initial_q)
    start_keys = {(environment.start_state, action) for action in actions}
    top_left, top_right = 90, 830
    top_top, top_bottom = 95, 295
    bottom_top, bottom_bottom = 380, 520
    plotted = [0, *range(1, len(records) + 1, max(1, len(records) // 300)), len(records)]
    plotted = sorted(set(plotted))
    q_values = list(result.initial_q.values())
    q_values += [value for record in records for value in record.q_values.values()]
    q_min, q_max = min(q_values), max(q_values)
    if q_min == q_max:
        q_min -= 1
        q_max += 1
    else:
        margin = (q_max - q_min) * 0.08
        q_min -= margin
        q_max += margin
    changes = [record.max_abs_q_change for record in records]
    change_max = max(changes) or 1.0
    x_at = lambda episode: top_left + (top_right - top_left) * episode / len(records)
    y_q = lambda value: top_bottom - (value - q_min) / (q_max - q_min) * (top_bottom - top_top)
    y_change = lambda value: bottom_bottom - value / change_max * (bottom_bottom - bottom_top)
    colors = ("#007f73", "#b8543c", "#6c51a3", "#9a7014", "#2865a5")
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PLOT_WIDTH}" height="{PLOT_HEIGHT}" viewBox="0 0 {PLOT_WIDTH} {PLOT_HEIGHT}" role="img" aria-labelledby="title desc">',
        f'<title id="title">Q-table convergence: {html.escape(environment.start_state)} to {html.escape(environment.target_state)}</title>',
        '<desc id="desc">Upper panel: all legal Q-table values, with start-state actions highlighted, initially and after each episode. Lower panel: maximum absolute Q-table entry change within each episode. All plotted values come from the saved convergence CSV.</desc>',
        '<rect width="900" height="600" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#263238">',
        f'<text x="90" y="38" font-size="22" font-weight="bold">Q-table convergence: {html.escape(environment.start_state)} to {html.escape(environment.target_state)}</text>',
        '<text x="90" y="67" font-size="14">All legal Q-values; start-state actions highlighted</text>',
        '<text x="90" y="356" font-size="14">Largest Q-value change across the full table per episode</text>',
    ]
    for top, bottom, minimum, maximum, mapping in (
        (top_top, top_bottom, q_min, q_max, y_q),
        (bottom_top, bottom_bottom, 0, change_max, y_change),
    ):
        for tick in range(5):
            value = minimum + (maximum - minimum) * tick / 4
            y = mapping(value)
            pieces.append(f'<line x1="{top_left}" y1="{y:.2f}" x2="{top_right}" y2="{y:.2f}" stroke="#e2e8e8"/>')
            pieces.append(f'<text x="80" y="{y + 4:.2f}" font-size="12" text-anchor="end">{value:.1f}</text>')
        pieces.append(f'<line x1="{top_left}" y1="{top}" x2="{top_left}" y2="{bottom}" stroke="#657275"/>')
        pieces.append(f'<line x1="{top_left}" y1="{bottom}" x2="{top_right}" y2="{bottom}" stroke="#657275"/>')
        for tick in range(5):
            episode = round(len(records) * tick / 4)
            x = x_at(episode)
            pieces.append(f'<text x="{x:.2f}" y="{bottom + 21}" font-size="12" text-anchor="middle">{episode}</text>')

    for state, action in all_keys:
        if (state, action) in start_keys:
            continue
        values = [result.initial_q[state, action]]
        values += [record.q_values[state, action] for record in records]
        points = [(x_at(i), y_q(values[i])) for i in plotted]
        pieces.append(f'<polyline points="{_polyline(points)}" fill="none" stroke="#67767a" stroke-opacity="0.28" stroke-width="1.3"><title>Q({html.escape(state)}, {html.escape(action)})</title></polyline>')

    for index, action in enumerate(actions):
        color = colors[index % len(colors)]
        values = [result.initial_q[environment.start_state, action]]
        values += [record.q_values[environment.start_state, action] for record in records]
        points = [(x_at(i), y_q(values[i])) for i in plotted]
        pieces.append(f'<polyline points="{_polyline(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_x = 95 + index * 145
        pieces.append(f'<line x1="{legend_x}" y1="327" x2="{legend_x + 22}" y2="327" stroke="{color}" stroke-width="3"/>')
        pieces.append(f'<text x="{legend_x + 28}" y="331" font-size="13">Q({html.escape(environment.start_state)}, {html.escape(action)})</text>')
    legend_x = 95 + len(actions) * 145
    pieces.append(f'<line x1="{legend_x}" y1="327" x2="{legend_x + 22}" y2="327" stroke="#67767a" stroke-opacity="0.45" stroke-width="2"/>')
    pieces.append(f'<text x="{legend_x + 28}" y="331" font-size="13">Other legal actions</text>')

    change_points = [(x_at(i), y_change(changes[i - 1])) for i in plotted if i > 0]
    pieces.append(f'<polyline points="{_polyline(change_points)}" fill="none" stroke="#3f65a4" stroke-width="1.6"/>')
    pieces.append('<text x="460" y="573" text-anchor="middle" font-size="14">Episode</text>')
    pieces.append('</g></svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def draw_case_study_heatmap(path: Path, environment: Environment, result) -> None:
    """Draw the final C-to-F Q table as an annotated state/action heatmap."""
    width, height = 900, 640
    left, top = 170, 130
    cell_width, cell_height = 96, 58
    columns = list(environment.states)
    q_values = list(result.final_q.values())
    value_min, value_max = 0.0, max(100.0, max(q_values, default=100.0))

    def color(value: float) -> str:
        ratio = max(0.0, min(1.0, (value - value_min) / (value_max - value_min)))
        red = round(242 - 223 * ratio)
        green = round(247 - 119 * ratio)
        blue = round(244 - 133 * ratio)
        return f"rgb({red},{green},{blue})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Annotated final Q-value heatmap for the C to F case study</title>',
        '<desc id="desc">Rows are current states and columns are possible next states. Darker cells have larger learned Q values. Gray cells are illegal actions or the terminal state. Orange outlines mark greedy actions. The highlighted route is C to B to F.</desc>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#263238">',
        '<text x="70" y="38" font-size="22" font-weight="bold">Case study: how C reaches F</text>',
        '<text x="70" y="66" font-size="14">Final Q values after 10,000 episodes; rows are current states and columns are next states</text>',
        '<text x="70" y="91" font-size="13">Darker cells mean a higher expected discounted return. Orange outlines mark the greedy action in each non-terminal row.</text>',
        '<text x="115" y="145" font-size="14" text-anchor="middle" font-weight="bold">State</text>',
    ]
    for index, state in enumerate(columns):
        x = left + index * cell_width + cell_width / 2
        parts.append(f'<text x="{x:.1f}" y="145" font-size="14" text-anchor="middle" font-weight="bold">next {html.escape(state)}</text>')

    for row_index, state in enumerate(environment.states):
        y = top + row_index * cell_height
        parts.append(f'<text x="115" y="{y + 35}" font-size="14" text-anchor="middle" font-weight="bold">{html.escape(state)}</text>')
        legal_actions = environment.available_actions(state)
        best_value = max((result.final_q[state, action] for action in legal_actions), default=None)
        for col_index, action in enumerate(columns):
            x = left + col_index * cell_width
            key = (state, action)
            if key in result.final_q:
                value = result.final_q[key]
                fill = color(value)
                text_color = "white" if value >= 58 else "#263238"
                best = best_value is not None and abs(value - best_value) <= 1e-6
                stroke = "#b8543c" if best else "#ffffff"
                stroke_width = 3 if best else 1
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')
                parts.append(f'<text x="{x + cell_width / 2}" y="{y + 25}" font-size="15" text-anchor="middle" fill="{text_color}" font-weight="{"bold" if best else "normal"}">{value:.1f}</text>')
                parts.append(f'<text x="{x + cell_width / 2}" y="{y + 45}" font-size="11" text-anchor="middle" fill="{text_color}">{html.escape(state)} → {html.escape(action)}</text>')
                parts.append(f'<title>Q({html.escape(state)}, {html.escape(action)}) = {value:.6f}'+("; greedy action" if best else "")+'</title>')
            else:
                label = "terminal" if state == environment.target_state else "-"
                parts.append(f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="#edf1f1" stroke="#ffffff" stroke-width="1"/>')
                parts.append(f'<text x="{x + cell_width / 2}" y="{y + 34}" font-size="13" text-anchor="middle" fill="#667477">{label}</text>')

    legend_y = top + len(environment.states) * cell_height + 36
    parts.append(f'<text x="{left}" y="{legend_y}" font-size="13">Q-value scale</text>')
    for index, value in enumerate((0, 25, 50, 75, 100)):
        x = left + 104 + index * 74
        parts.append(f'<rect x="{x}" y="{legend_y - 16}" width="45" height="15" fill="{color(value)}" stroke="#ffffff"/>')
        parts.append(f'<text x="{x + 22.5}" y="{legend_y + 12}" font-size="11" text-anchor="middle">{value}</text>')
    parts.append(f'<rect x="{left + 505}" y="{legend_y - 16}" width="45" height="15" fill="#edf1f1" stroke="#ffffff"/>')
    parts.append(f'<text x="{left + 560}" y="{legend_y - 4}" font-size="12">illegal / terminal</text>')
    parts.append(f'<rect x="{left + 635}" y="{legend_y - 16}" width="35" height="15" fill="white" stroke="#b8543c" stroke-width="3"/>')
    parts.append(f'<text x="{left + 685}" y="{legend_y - 4}" font-size="12">greedy</text>')
    parts.append('<text x="70" y="610" font-size="14"><tspan font-weight="bold">Reading the case:</tspan> Q(C,B) = 89.0 is larger than Q(C,D) = 79.1, then Q(B,F) = 100.0. The greedy route is therefore C → B → F.</text>')
    parts.append('</g></svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def print_summary(environment: Environment, result, output_dir: Path, plot_path: Path) -> None:
    successes = sum(record.reached_target for record in result.records)
    final = result.records[-1]
    print(f"{environment.start_state} -> {environment.target_state}: "
          f"{successes}/{len(result.records)} episodes reached target")
    print(f"  Last episode maximum Q-table change: {final.max_abs_q_change:.6g}")
    print("  Final Q-table (state -> action: value, visits):")
    for state in environment.states:
        for action in environment.available_actions(state):
            key = (state, action)
            print(f"    {state} -> {action}: {result.final_q[key]:.6f}, {result.visits[key]}")
    print(f"  CSV: {output_dir}")
    print(f"  Plot: {plot_path}")


def update_report(environment: Environment, result, plot_path: Path, section: str) -> None:
    report_path = ROOT / "report/index.html"
    document = report_path.read_text(encoding="utf-8")
    start_marker = f"<!-- RESULTS_{section}_START -->"
    end_marker = f"<!-- RESULTS_{section}_END -->"
    if document.count(start_marker) != 1 or document.count(end_marker) != 1:
        raise ValueError(f"report is missing unique {section} result markers")

    successful = sum(record.reached_target for record in result.records)
    tail = result.records[-min(500, len(result.records)):]
    largest_recent_change = max(record.max_abs_q_change for record in tail)
    least_visits = min(result.visits.values())
    title = "Original: C to F" if section == "BASELINE" else "Changed: A to C"
    relative_plot = plot_path.relative_to(report_path.parent).as_posix()
    lines = [
        f'    <h3>{title}</h3>',
        f'    <p>{successful:,} of {len(result.records):,} episodes reached the target. '
        f'The largest single-episode Q-table change in the last {len(tail):,} episodes '
        f'was {largest_recent_change:.6g}; the least-visited legal action was used '
        f'{least_visits:,} times. These observations do not establish a convergence guarantee.</p>',
        '    <figure>',
        f'      <img src="{html.escape(relative_plot, quote=True)}" '
        f'alt="Convergence plot for {html.escape(environment.start_state)} to '
        f'{html.escape(environment.target_state)}: start-state Q-values and '
        'largest whole-table change by episode">',
        '      <figcaption>Upper panel: all legal Q-values, with start-state actions highlighted; lower panel: maximum '
        'absolute Q-table change per episode. Full numerical series is saved in '
        f'<code>results/{section.lower() if section == "BASELINE" else "start_A_goal_C"}/convergence.csv</code>.</figcaption>',
        '    </figure>',
        '    <div class="table-wrap"><table>',
        '      <thead><tr><th scope="col">State</th><th scope="col">Action / next state</th>'
        '<th scope="col">Initial Q</th><th scope="col">Final Q</th>'
        '<th scope="col">Visits</th></tr></thead>',
        '      <tbody>',
    ]
    for state in environment.states:
        for action in environment.available_actions(state):
            key = (state, action)
            lines.append(
                f'        <tr><th scope="row">{html.escape(state)}</th>'
                f'<td>{html.escape(action)}</td>'
                f'<td>{result.initial_q[key]:.6f}</td>'
                f'<td>{result.final_q[key]:.6f}</td>'
                f'<td>{result.visits[key]}</td></tr>'
            )
    lines.extend(('      </tbody>', '    </table></div>'))
    before, rest = document.split(start_marker, 1)
    _, after = rest.split(end_marker, 1)
    report_path.write_text(
        before + start_marker + "\n" + "\n".join(lines) + "\n    " + end_marker + after,
        encoding="utf-8",
    )


def run_config(config_path: Path, start: str | None, target: str | None,
               output_dir: Path | None, plot_path: Path | None) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if start is not None:
        config["start_state"] = start
    if target is not None:
        config["target_state"] = target
    environment = Environment.from_config(config)
    settings = TrainingSettings.from_config(config)
    result = train(environment, settings)

    suffix = f"_{environment.start_state}_to_{environment.target_state}" if start or target else ""
    name = config_path.stem + suffix
    output_dir = output_dir or ROOT / "results" / name
    plot_path = plot_path or ROOT / "report/assets" / f"{name}_convergence.svg"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    write_q_table(output_dir / "q_initial.csv", environment, result.initial_q,
                  {key: 0 for key in result.initial_q})
    write_q_table(output_dir / "q_final.csv", environment, result.final_q, result.visits)
    write_convergence(output_dir / "convergence.csv", environment, result.records)
    draw_plot(plot_path, environment, result)
    if (start is None and target is None
            and config_path.resolve() in (path.resolve() for path in DEFAULT_CONFIGS)
            and output_dir == ROOT / "results" / config_path.stem
            and plot_path == ROOT / "report/assets" / f"{config_path.stem}_convergence.svg"):
        if config_path.stem == "baseline":
            draw_case_study_heatmap(ROOT / "report/assets/case_study_q_heatmap.svg", environment, result)
        section = "BASELINE" if config_path.stem == "baseline" else "CHANGED"
        update_report(environment, result, plot_path, section)
    print_summary(environment, result, output_dir, plot_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", help="JSON config; repeatable. Defaults to both supplied configs")
    parser.add_argument("--start", help="Override start state for one config")
    parser.add_argument("--target", help="Override target state for one config")
    parser.add_argument("--output-dir", type=Path, help="Output CSV directory for one config")
    parser.add_argument("--plot", type=Path, help="Output SVG path for one config")
    args = parser.parse_args()
    configs = args.config or DEFAULT_CONFIGS
    if len(configs) != 1 and any((args.start, args.target, args.output_dir, args.plot)):
        parser.error("overrides and output paths require exactly one --config")
    for path in configs:
        run_config(path, args.start, args.target, args.output_dir, args.plot)


if __name__ == "__main__":
    main()
