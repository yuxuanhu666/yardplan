from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def plot_convergence_report(
    result_or_metrics: Any,
    output_path: str | Path = "outputs/convergence.svg",
    *,
    title: str = "Yard Planning Algorithm Progress",
    dpi: int = 220,
) -> Path:
    metrics = _extract_metrics(result_or_metrics)
    stage1 = list(metrics.get("stage1_convergence") or [])
    stage2 = list(metrics.get("stage2_progress") or [])

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        output = Path(output_path)
        svg_output = output if output.suffix.lower() == ".svg" else output.with_suffix(".svg")
        return _plot_svg(stage1, stage2, svg_output, title=title)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#f8fafc",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.0))
    fig.suptitle(title, fontsize=18, fontweight="bold", color="#0f172a", y=0.985)

    _plot_stage1_cost_matplotlib(axes[0], stage1)
    _plot_sa_acceptance_matplotlib(axes[1], stage1)
    _plot_stage2_progress_matplotlib(axes[2], stage2)

    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.965))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_stage1_cost_matplotlib(ax: Any, rows: Sequence[Dict[str, Any]]) -> None:
    ax.set_title("Stage1 LNS Cost Convergence", loc="left", fontweight="bold")
    if not rows:
        _empty_axis(ax, "No Stage1 convergence history")
        return
    x = [_as_float(row.get("iteration")) for row in rows]
    best = [_as_float(row.get("bestCost")) for row in rows]
    current = [_as_float(row.get("currentCost")) for row in rows]
    ax.plot(x, current, color="#94a3b8", linewidth=1.8, label="current_cost")
    ax.plot(x, best, color="#2563eb", linewidth=2.8, label="best_cost")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost (lower is better)")
    ax.grid(True, color="#dbe3ef", linewidth=0.9)
    _style_numeric_axes(ax, integer_x=True)
    _annotate_last_value_matplotlib(ax, x, current, "current", "#64748b", dy=-16)
    _annotate_last_value_matplotlib(ax, x, best, "best", "#2563eb", dy=10)
    ax.legend(loc="upper right", frameon=False)


def _plot_sa_acceptance_matplotlib(ax: Any, rows: Sequence[Dict[str, Any]]) -> None:
    ax.set_title("Simulated Annealing Acceptance", loc="left", fontweight="bold")
    if len(rows) <= 1:
        _empty_axis(ax, "No SA iteration history")
        return
    iter_rows = rows[1:]
    x = [_as_float(row.get("iteration")) for row in iter_rows]
    accepted = [1.0 if row.get("accepted") else 0.0 for row in iter_rows]
    colors = ["#16a34a" if value > 0.5 else "#ef4444" for value in accepted]
    ax.scatter(x, accepted, c=colors, s=32, alpha=0.9, label="accepted")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0 rejected", "1 accepted"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SA acceptance")
    ax.set_ylim(-0.25, 1.25)
    ax.grid(True, color="#dbe3ef", linewidth=0.9)
    _style_numeric_axes(ax, integer_x=True, numeric_y=False)

    temp = [_as_float(row.get("temperature")) for row in iter_rows]
    if temp and max(temp) > 0:
        twin = ax.twinx()
        twin.plot(x, temp, color="#f59e0b", linewidth=2.0, label="temperature")
        twin.set_ylabel("Temperature")
        twin.tick_params(axis="y", colors="#b45309")
        _style_numeric_axes(twin, integer_x=True, color="#b45309")
        _annotate_last_value_matplotlib(twin, x, temp, "temperature", "#b45309", dy=10)


def _plot_stage2_progress_matplotlib(ax: Any, rows: Sequence[Dict[str, Any]]) -> None:
    ax.set_title("Stage2 Greedy Placement Progress", loc="left", fontweight="bold")
    if not rows:
        _empty_axis(ax, "No Stage2 greedy placement history")
        return
    x = [_as_float(row.get("step")) for row in rows]
    placed = [_as_float(row.get("globalPlaced")) for row in rows]
    remaining = [_as_float(row.get("globalRemaining")) for row in rows]
    ax.plot(x, placed, color="#0f766e", linewidth=2.8, label="placed")
    ax.plot(x, remaining, color="#dc2626", linewidth=2.2, label="remaining")
    ax.fill_between(x, placed, color="#99f6e4", alpha=0.32)
    ax.set_xlabel("Greedy placement step")
    ax.set_ylabel("Container count")
    ax.grid(True, color="#dbe3ef", linewidth=0.9)
    _style_numeric_axes(ax, integer_x=True)
    _annotate_last_value_matplotlib(ax, x, placed, "placed", "#0f766e", dy=10)
    _annotate_last_value_matplotlib(ax, x, remaining, "remaining", "#dc2626", dy=-16)
    ax.legend(loc="center right", frameon=False)


def _style_numeric_axes(
    ax: Any,
    *,
    integer_x: bool = False,
    numeric_y: bool = True,
    color: str = "#475569",
) -> None:
    try:
        from matplotlib.ticker import MaxNLocator, ScalarFormatter

        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=integer_x))
        if numeric_y:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
    except Exception:
        pass
    ax.tick_params(axis="both", labelsize=9, colors=color)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")


def _annotate_last_value_matplotlib(
    ax: Any,
    xs: Sequence[float],
    ys: Sequence[float],
    label: str,
    color: str,
    *,
    dy: int = 0,
) -> None:
    if not xs or not ys:
        return
    ax.annotate(
        f"{label}: {_fmt_axis_tick(ys[-1])}",
        xy=(xs[-1], ys[-1]),
        xytext=(8, dy),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.85},
        clip_on=False,
    )


def _empty_axis(ax: Any, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="#64748b")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _plot_svg(
    stage1: Sequence[Dict[str, Any]],
    stage2: Sequence[Dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    width = 1080
    height = 980
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        (112, 145, 860, 220, "Stage1 LNS Cost Convergence"),
        (112, 410, 860, 190, "Simulated Annealing Acceptance"),
        (112, 650, 860, 230, "Stage2 Greedy Placement Progress"),
    ]

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<rect width='100%' height='100%' fill='white'/>",
        "<rect x='36' y='38' width='1008' height='900' rx='26' fill='#f8fafc' stroke='#e2e8f0'/>",
        f"<text x='{width / 2:.0f}' y='88' text-anchor='middle' font-family='DejaVu Sans, Arial, sans-serif' font-size='30' font-weight='800' fill='#0f172a'>{escape(title)}</text>",
    ]

    parts.extend(_svg_stage1_cost(stage1, panels[0]))
    parts.extend(_svg_sa_acceptance(stage1, panels[1]))
    parts.extend(_svg_stage2_progress(stage2, panels[2]))
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


def _svg_stage1_cost(rows: Sequence[Dict[str, Any]], panel: Tuple[int, int, int, int, str]) -> List[str]:
    x, y, w, h, label = panel
    if not rows:
        return _svg_empty_panel(panel, "No Stage1 convergence history")
    xs = [_as_float(row.get("iteration")) for row in rows]
    best = [_as_float(row.get("bestCost")) for row in rows]
    current = [_as_float(row.get("currentCost")) for row in rows]
    ymin, ymax = _domain(best + current)
    xmin, xmax = min(xs), max(xs)
    return (
        _svg_panel_base(
            panel,
            x_label="Iteration",
            y_label="Objective cost (lower is better)",
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        + [_svg_polyline(xs, current, x, y, w, h, xmin, xmax, ymin, ymax, "#94a3b8", 2)]
        + [_svg_polyline(xs, best, x, y, w, h, xmin, xmax, ymin, ymax, "#2563eb", 4)]
        + _svg_value_label(xs, current, x, y, w, h, xmin, xmax, ymin, ymax, "current", "#64748b", dy=-12)
        + _svg_value_label(xs, best, x, y, w, h, xmin, xmax, ymin, ymax, "best", "#2563eb", dy=16)
        + _svg_legend(x + w - 210, y + 20, [("current_cost", "#94a3b8"), ("best_cost", "#2563eb")])
    )


def _svg_sa_acceptance(rows: Sequence[Dict[str, Any]], panel: Tuple[int, int, int, int, str]) -> List[str]:
    x, y, w, h, label = panel
    if len(rows) <= 1:
        return _svg_empty_panel(panel, "No SA iteration history")
    iter_rows = list(rows[1:])
    xs = [_as_float(row.get("iteration")) for row in iter_rows]
    temp = [_as_float(row.get("temperature")) for row in iter_rows]
    xmin, xmax = min(xs), max(xs)
    parts = _svg_panel_base(
        panel,
        x_label="Iteration",
        y_label="SA acceptance (0 rejected, 1 accepted)",
        xmin=xmin,
        xmax=xmax,
        ymin=0.0,
        ymax=1.0,
        y_ticks=[0.0, 1.0],
        y_tick_labels=["0", "1"],
    )
    for row, x_value in zip(iter_rows, xs):
        accepted = bool(row.get("accepted"))
        px, py = _map_point(x_value, 1.0 if accepted else 0.0, x, y, w, h, xmin, xmax, 0.0, 1.0)
        color = "#16a34a" if accepted else "#ef4444"
        parts.append(f"<circle cx='{px:.2f}' cy='{py:.2f}' r='5.5' fill='{color}' opacity='0.9'/>")
    if temp and max(temp) > min(temp):
        temp_min, temp_max = min(temp), max(temp)
        parts.append(_svg_polyline(xs, temp, x, y, w, h, xmin, xmax, temp_min, temp_max, "#f59e0b", 3))
        parts.extend(_svg_right_axis(panel, temp_min, temp_max, "Temperature", "#b45309"))
        parts.extend(_svg_value_label(xs, temp, x, y, w, h, xmin, xmax, temp_min, temp_max, "temp", "#b45309", dy=-12))
    parts.extend(_svg_legend(x + w - 250, y + 18, [("accepted/rejected", "#16a34a"), ("temperature", "#f59e0b")]))
    return parts


def _svg_stage2_progress(rows: Sequence[Dict[str, Any]], panel: Tuple[int, int, int, int, str]) -> List[str]:
    x, y, w, h, label = panel
    if not rows:
        return _svg_empty_panel(panel, "No Stage2 greedy placement history")
    xs = [_as_float(row.get("step")) for row in rows]
    placed = [_as_float(row.get("globalPlaced")) for row in rows]
    remaining = [_as_float(row.get("globalRemaining")) for row in rows]
    ymin, ymax = _domain(placed + remaining + [_as_float(rows[-1].get("globalDemand"))])
    xmin, xmax = min(xs), max(xs)
    return (
        _svg_panel_base(
            panel,
            x_label="Greedy placement step",
            y_label="Container count",
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )
        + [_svg_polyline(xs, placed, x, y, w, h, xmin, xmax, ymin, ymax, "#0f766e", 4)]
        + [_svg_polyline(xs, remaining, x, y, w, h, xmin, xmax, ymin, ymax, "#dc2626", 3)]
        + _svg_value_label(xs, placed, x, y, w, h, xmin, xmax, ymin, ymax, "placed", "#0f766e", dy=-12)
        + _svg_value_label(xs, remaining, x, y, w, h, xmin, xmax, ymin, ymax, "remaining", "#dc2626", dy=16)
        + _svg_legend(x + w - 180, y + 20, [("placed", "#0f766e"), ("remaining", "#dc2626")])
    )


def _svg_panel_base(
    panel: Tuple[int, int, int, int, str],
    *,
    x_label: str,
    y_label: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    x_ticks: Optional[Sequence[float]] = None,
    y_ticks: Optional[Sequence[float]] = None,
    y_tick_labels: Optional[Sequence[str]] = None,
) -> List[str]:
    x, y, w, h, label = panel
    x_ticks = list(x_ticks or _nice_ticks(xmin, xmax, 5))
    y_ticks = list(y_ticks or _nice_ticks(ymin, ymax, 5))
    y_tick_labels = list(y_tick_labels or [_fmt_axis_tick(value) for value in y_ticks])
    parts = [
        f"<text x='{x}' y='{y - 18}' font-family='DejaVu Sans, Arial, sans-serif' font-size='19' font-weight='800' fill='#0f172a'>{escape(label)}</text>",
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='14' fill='white' stroke='#e2e8f0'/>",
        f"<line x1='{x}' y1='{y + h}' x2='{x + w}' y2='{y + h}' stroke='#334155' stroke-width='1.4'/>",
        f"<line x1='{x}' y1='{y}' x2='{x}' y2='{y + h}' stroke='#334155' stroke-width='1.4'/>",
    ]
    for tick in y_ticks:
        _px, yy = _map_point(xmin, tick, x, y, w, h, xmin, xmax, ymin, ymax)
        parts.append(f"<line x1='{x}' y1='{yy:.2f}' x2='{x + w}' y2='{yy:.2f}' stroke='#e2e8f0' stroke-width='1'/>")
        parts.append(f"<line x1='{x - 5}' y1='{yy:.2f}' x2='{x}' y2='{yy:.2f}' stroke='#334155' stroke-width='1'/>")
    for tick, tick_label in zip(y_ticks, y_tick_labels):
        _px, yy = _map_point(xmin, tick, x, y, w, h, xmin, xmax, ymin, ymax)
        parts.append(f"<text x='{x - 11}' y='{yy + 4:.2f}' text-anchor='end' font-family='DejaVu Sans, Arial, sans-serif' font-size='12' fill='#475569'>{escape(tick_label)}</text>")

    for tick in x_ticks:
        px, _py = _map_point(tick, ymin, x, y, w, h, xmin, xmax, ymin, ymax)
        parts.append(f"<line x1='{px:.2f}' y1='{y}' x2='{px:.2f}' y2='{y + h}' stroke='#edf2f7' stroke-width='1'/>")
        parts.append(f"<line x1='{px:.2f}' y1='{y + h}' x2='{px:.2f}' y2='{y + h + 5}' stroke='#334155' stroke-width='1'/>")
        parts.append(f"<text x='{px:.2f}' y='{y + h + 20}' text-anchor='middle' font-family='DejaVu Sans, Arial, sans-serif' font-size='12' fill='#475569'>{escape(_fmt_axis_tick(tick))}</text>")

    parts.append(f"<text x='{x + w / 2:.2f}' y='{y + h + 42}' text-anchor='middle' font-family='DejaVu Sans, Arial, sans-serif' font-size='14' font-weight='700' fill='#334155'>{escape(x_label)}</text>")
    parts.append(
        f"<text x='{x - 76}' y='{y + h / 2:.2f}' text-anchor='middle' "
        "font-family='DejaVu Sans, Arial, sans-serif' font-size='14' font-weight='700' "
        f"fill='#334155' transform='rotate(-90 {x - 76} {y + h / 2:.2f})'>{escape(y_label)}</text>"
    )
    return parts


def _svg_empty_panel(panel: Tuple[int, int, int, int, str], text: str) -> List[str]:
    x, y, w, h, _label = panel
    return _svg_panel_base(
        panel,
        x_label="",
        y_label="",
        xmin=0.0,
        xmax=1.0,
        ymin=0.0,
        ymax=1.0,
    ) + [
        f"<text x='{x + w / 2:.2f}' y='{y + h / 2:.2f}' text-anchor='middle' font-family='DejaVu Sans, Arial, sans-serif' font-size='16' fill='#64748b'>{escape(text)}</text>"
    ]


def _svg_polyline(
    xs: Sequence[float],
    ys: Sequence[float],
    x: float,
    y: float,
    w: float,
    h: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    color: str,
    stroke_width: int,
) -> str:
    points = [
        _map_point(x_value, y_value, x, y, w, h, xmin, xmax, ymin, ymax)
        for x_value, y_value in zip(xs, ys)
    ]
    text = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
    return f"<polyline points='{text}' fill='none' stroke='{color}' stroke-width='{stroke_width}' stroke-linecap='round' stroke-linejoin='round'/>"


def _svg_value_label(
    xs: Sequence[float],
    ys: Sequence[float],
    x: float,
    y: float,
    w: float,
    h: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    label: str,
    color: str,
    *,
    dy: float = 0.0,
) -> List[str]:
    if not xs or not ys:
        return []
    px, py = _map_point(xs[-1], ys[-1], x, y, w, h, xmin, xmax, ymin, ymax)
    text = f"{label}: {_fmt_axis_tick(ys[-1])}"
    text_width = max(58, len(text) * 7 + 12)
    tx = px + 10
    if tx + text_width > x + w - 4:
        tx = x + w - text_width - 4
    tx = max(x + 4, tx)
    ty = min(max(py + dy, y + 16), y + h - 8)
    return [
        f"<rect x='{tx - 6:.2f}' y='{ty - 14:.2f}' width='{text_width:.2f}' height='19' rx='5' fill='white' fill-opacity='0.88' stroke='{color}' stroke-opacity='0.65'/>",
        f"<text x='{tx:.2f}' y='{ty:.2f}' font-family='DejaVu Sans, Arial, sans-serif' font-size='12' font-weight='700' fill='{color}'>{escape(text)}</text>",
    ]


def _map_point(
    x_value: float,
    y_value: float,
    x: float,
    y: float,
    w: float,
    h: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Tuple[float, float]:
    xr = 0.0 if xmax <= xmin else (x_value - xmin) / (xmax - xmin)
    yr = 0.0 if ymax <= ymin else (y_value - ymin) / (ymax - ymin)
    return x + xr * w, y + h - yr * h


def _svg_legend(x: float, y: float, entries: Sequence[Tuple[str, str]]) -> List[str]:
    parts = []
    for index, (label, color) in enumerate(entries):
        yy = y + index * 24
        parts.append(f"<line x1='{x}' y1='{yy}' x2='{x + 30}' y2='{yy}' stroke='{color}' stroke-width='4'/>")
        parts.append(f"<text x='{x + 40}' y='{yy + 5}' font-family='DejaVu Sans, Arial, sans-serif' font-size='13' fill='#334155'>{escape(label)}</text>")
    return parts


def _svg_right_axis(
    panel: Tuple[int, int, int, int, str],
    ymin: float,
    ymax: float,
    label: str,
    color: str,
) -> List[str]:
    x, y, w, h, _label = panel
    axis_x = x + w
    ticks = _nice_ticks(ymin, ymax, 4)
    parts = [
        f"<line x1='{axis_x}' y1='{y}' x2='{axis_x}' y2='{y + h}' stroke='{color}' stroke-width='1.2'/>"
    ]
    for tick in ticks:
        _px, yy = _map_point(0.0, tick, x, y, w, h, 0.0, 1.0, ymin, ymax)
        parts.append(f"<line x1='{axis_x}' y1='{yy:.2f}' x2='{axis_x + 5}' y2='{yy:.2f}' stroke='{color}' stroke-width='1'/>")
        parts.append(f"<text x='{axis_x + 10}' y='{yy + 4:.2f}' font-family='DejaVu Sans, Arial, sans-serif' font-size='12' fill='{color}'>{escape(_fmt_axis_tick(tick))}</text>")
    parts.append(
        f"<text x='{axis_x + 72}' y='{y + h / 2:.2f}' text-anchor='middle' "
        "font-family='DejaVu Sans, Arial, sans-serif' font-size='14' font-weight='700' "
        f"fill='{color}' transform='rotate(90 {axis_x + 72} {y + h / 2:.2f})'>{escape(label)}</text>"
    )
    return parts


def _domain(values: Iterable[float]) -> Tuple[float, float]:
    values = list(values)
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if high <= low:
        pad = max(1.0, abs(high) * 0.1)
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def _nice_ticks(low: float, high: float, target_count: int = 5) -> List[float]:
    if target_count <= 0:
        target_count = 5
    low = float(low)
    high = float(high)
    if not (high > low):
        return [low]

    span = high - low
    raw_step = span / max(1, target_count - 1)
    if raw_step <= 0:
        return [low, high]

    magnitude = 10 ** int(max(0, math.floor(math.log10(abs(raw_step)))))
    normalized = raw_step / magnitude
    if normalized < 1.5:
        step = 1.0 * magnitude
    elif normalized < 3.0:
        step = 2.0 * magnitude
    elif normalized < 7.0:
        step = 5.0 * magnitude
    else:
        step = 10.0 * magnitude

    start = math.floor(low / step) * step
    end = math.ceil(high / step) * step
    ticks = []
    value = start
    guard = 0
    while value <= end + step * 0.5 and guard < 100:
        ticks.append(round(value, 10))
        value += step
        guard += 1
    if len(ticks) < 2:
        ticks = [low, high]
    return ticks


def _fmt_axis_tick(value: float) -> str:
    value = float(value)
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 1000:
        if abs(value - round(value)) < 1e-6:
            return f"{int(round(value)):,}"
        return f"{value:,.1f}"
    if abs(value) < 0.01:
        return f"{value:.1e}"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _extract_metrics(result_or_metrics: Any) -> Dict[str, Any]:
    metrics = getattr(result_or_metrics, "metrics", None)
    if isinstance(metrics, dict):
        return metrics
    if isinstance(result_or_metrics, dict) and "metrics" in result_or_metrics:
        nested = result_or_metrics.get("metrics")
        if isinstance(nested, dict):
            return nested
    if isinstance(result_or_metrics, dict):
        return result_or_metrics
    return {}


def _load_metrics_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return _extract_metrics(data)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Stage1 convergence and Stage2 progress charts.")
    parser.add_argument("--metrics-json", type=str, help="JSON file containing result metrics.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--line-key", dest="line_key", type=int, help="Service line key.")
    target.add_argument("--vessel-key", dest="vessel_key", type=int, default=15623707)
    parser.add_argument("--type", dest="plan_type", type=int, default=1)
    parser.add_argument("--start", type=str)
    parser.add_argument("--end", type=str)
    parser.add_argument("--output", type=str, default="outputs/convergence.svg")
    parser.add_argument("--title", type=str, default="Yard Planning Algorithm Progress")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    if args.metrics_json:
        metrics = _load_metrics_json(args.metrics_json)
    else:
        from yardplan import run_plan

        kwargs: Dict[str, Any] = {
            "type": args.plan_type,
            "save_visualization": False,
            "print_score": False,
        }
        if args.line_key is not None:
            kwargs["line_keys"] = args.line_key
        else:
            kwargs["vessel_key"] = args.vessel_key
        if args.start:
            kwargs["plan_start_time"] = datetime.fromisoformat(args.start)
        if args.end:
            kwargs["plan_end_time"] = datetime.fromisoformat(args.end)
        result = run_plan(**kwargs)
        metrics = getattr(result, "metrics", {})

    output = plot_convergence_report(metrics, args.output, title=args.title, dpi=args.dpi)
    print(f"Convergence chart saved to: {output}")


if __name__ == "__main__":
    main()
