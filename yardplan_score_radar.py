from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ITEM_ORDER: Sequence[Tuple[str, str]] = (
    ("transport_distance", "Transport\nDistance"),
    ("demand_satisfaction", "Demand\nSatisfaction"),
    ("business_dispersion", "Business\nDispersion"),
    ("area_peak_staggering", "Peak\nStaggering"),
    ("bay_quality", "Bay\nQuality"),
)


def plot_score_radar(
    score: Dict[str, Any],
    output_path: str | Path = "outputs/score_radar.png",
    *,
    title: str = "Yard Plan Score Radar",
    dpi: int = 220,
) -> Path:
    """Plot the five 10-point score items as a radar chart."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        output = Path(output_path)
        svg_output = output if output.suffix.lower() == ".svg" else output.with_suffix(".svg")
        return _plot_score_radar_svg(score, svg_output, title=title)

    values = _score_values(score)
    labels = [label for _key, label in ITEM_ORDER]
    closed_values = values + values[:1]
    angles = [2.0 * math.pi * idx / len(values) for idx in range(len(values))]
    closed_angles = angles + angles[:1]

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

    fig = plt.figure(figsize=(8.2, 7.2))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 10)
    ax.spines["polar"].set_color("#cbd5e1")
    ax.spines["polar"].set_linewidth(1.1)
    ax.grid(color="#cbd5e1", linewidth=0.9, alpha=0.9)

    band_angles = [2.0 * math.pi * idx / 240 for idx in range(241)]
    for radius, color, alpha in (
        (10, "#dbeafe", 0.34),
        (8, "#e0f2fe", 0.34),
        (6, "#ecfeff", 0.44),
        (4, "#f8fafc", 0.88),
    ):
        ax.fill(band_angles, [radius] * len(band_angles), color=color, alpha=alpha, zorder=0)

    ax.plot(
        closed_angles,
        closed_values,
        color="#2563eb",
        linewidth=2.8,
        marker="o",
        markersize=7,
        markerfacecolor="#0f172a",
        markeredgecolor="white",
        markeredgewidth=1.8,
        zorder=4,
    )
    ax.fill(closed_angles, closed_values, color="#2563eb", alpha=0.22, zorder=3)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11, color="#0f172a", fontweight="semibold")
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=9, color="#64748b")
    ax.set_rlabel_position(90)

    for angle, value in zip(angles, values):
        label_radius = min(10.6, value + 0.75)
        ax.text(
            angle,
            label_radius,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=10,
            color="#1e293b",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#cbd5e1",
                "linewidth": 0.8,
                "alpha": 0.94,
            },
            zorder=5,
        )

    total_score = _as_float(score.get("totalScore"))
    max_score = _as_float(score.get("maxScore", 50.0)) or 50.0
    fig.suptitle(title, y=0.965, fontsize=17, fontweight="bold", color="#0f172a")
    fig.text(
        0.5,
        0.91,
        f"Total score: {total_score:.2f} / {max_score:.2f}",
        ha="center",
        va="center",
        fontsize=12,
        color="#475569",
    )
    fig.text(
        0.5,
        0.035,
        "Each axis is normalized to a 10-point indicator score.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#64748b",
    )

    fig.tight_layout(rect=(0.04, 0.06, 0.96, 0.9))
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_score_radar_svg(
    score: Dict[str, Any],
    output_path: str | Path,
    *,
    title: str,
) -> Path:
    values = _score_values(score)
    labels = [label.replace("\n", " ") for _key, label in ITEM_ORDER]
    width = 860
    height = 760
    cx = width / 2.0
    cy = 405.0
    radius = 230.0
    angles = [-math.pi / 2.0 + 2.0 * math.pi * idx / len(values) for idx in range(len(values))]

    def point(angle: float, value: float) -> Tuple[float, float]:
        r = radius * max(0.0, min(10.0, value)) / 10.0
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    def polygon(points: Sequence[Tuple[float, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_score = _as_float(score.get("totalScore"))
    max_score = _as_float(score.get("maxScore", 50.0)) or 50.0
    score_points = [point(angle, value) for angle, value in zip(angles, values)]
    axis_points = [point(angle, 10.0) for angle in angles]

    ring_parts: List[str] = []
    for ring_value, fill, stroke in (
        (10, "#dbeafe", "#bfdbfe"),
        (8, "#e0f2fe", "#bae6fd"),
        (6, "#ecfeff", "#cffafe"),
        (4, "#f8fafc", "#e2e8f0"),
        (2, "#ffffff", "#e2e8f0"),
    ):
        ring_points = [point(angle, ring_value) for angle in angles]
        ring_parts.append(
            f'<polygon points="{polygon(ring_points)}" fill="{fill}" fill-opacity="0.72" '
            f'stroke="{stroke}" stroke-width="1.2"/>'
        )

    axis_parts = []
    for angle, label, value in zip(angles, labels, values):
        x2, y2 = point(angle, 10.0)
        lx, ly = point(angle, 11.25)
        vx, vy = point(angle, min(10.65, value + 0.8))
        anchor = "middle"
        if lx < cx - 40:
            anchor = "end"
        elif lx > cx + 40:
            anchor = "start"
        axis_parts.append(
            f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            'stroke="#cbd5e1" stroke-width="1"/>'
        )
        axis_parts.append(
            f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="{anchor}" '
            'font-size="17" font-weight="700" fill="#0f172a">'
            f"{escape(label)}</text>"
        )
        axis_parts.append(
            f'<rect x="{vx - 22:.2f}" y="{vy - 15:.2f}" width="44" height="28" rx="8" '
            'fill="white" stroke="#cbd5e1" stroke-width="1" opacity="0.96"/>'
        )
        axis_parts.append(
            f'<text x="{vx:.2f}" y="{vy + 5:.2f}" text-anchor="middle" '
            'font-size="14" font-weight="700" fill="#1e293b">'
            f"{value:.1f}</text>"
        )

    tick_parts = []
    for tick in (2, 4, 6, 8, 10):
        tx, ty = point(-math.pi / 2.0, tick)
        tick_parts.append(
            f'<text x="{tx + 8:.2f}" y="{ty + 4:.2f}" font-size="12" fill="#64748b">{tick}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="scoreFill" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.34"/>
      <stop offset="100%" stop-color="#06b6d4" stop-opacity="0.24"/>
    </linearGradient>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="#1e293b" flood-opacity="0.12"/>
    </filter>
  </defs>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <rect x="42" y="42" width="776" height="666" rx="26" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="{cx:.2f}" y="92" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="28" font-weight="800" fill="#0f172a">{escape(title)}</text>
  <text x="{cx:.2f}" y="126" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="17" fill="#475569">Total score: {total_score:.2f} / {max_score:.2f}</text>
  <g font-family="DejaVu Sans, Arial, sans-serif">
    <g filter="url(#softShadow)">
      {''.join(ring_parts)}
    </g>
    {''.join(axis_parts)}
    {''.join(tick_parts)}
    <polygon points="{polygon(score_points)}" fill="url(#scoreFill)" stroke="#2563eb" stroke-width="4" stroke-linejoin="round"/>
    {''.join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#0f172a" stroke="#ffffff" stroke-width="3"/>' for x, y in score_points)}
  </g>
  <text x="{cx:.2f}" y="685" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" font-size="13" fill="#64748b">Each axis is normalized to a 10-point indicator score.</text>
</svg>
"""
    output.write_text(svg, encoding="utf-8")
    return output


def _score_values(score: Dict[str, Any]) -> List[float]:
    items = score.get("items") or {}
    values: List[float] = []
    for key, _label in ITEM_ORDER:
        item = items.get(key) or {}
        value = _as_float(item.get("score"))
        max_score = _as_float(item.get("maxScore", 10.0)) or 10.0
        values.append(max(0.0, min(10.0, 10.0 * value / max_score)))
    return values


def _load_score_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "items" in data:
        return data
    if isinstance(data.get("score"), dict):
        return data["score"]
    raise ValueError(f"No score object found in {path}")


def _run_plan_score(args: argparse.Namespace) -> Dict[str, Any]:
    from yardplan import run_plan
    from yardplan_score import score_yard_plan

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
    return getattr(result, "metrics", {}).get("score") or score_yard_plan(result)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a radar chart for yard-plan score items.")
    parser.add_argument(
        "--score-json",
        type=str,
        help="Path to a score JSON file. If omitted, the script runs yardplan first.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--line-key", dest="line_key", type=int, help="Service line key.")
    target.add_argument(
        "--vessel-key",
        dest="vessel_key",
        type=int,
        default=15623707,
        help="VesselVisit dbkey used when --score-json is omitted.",
    )
    parser.add_argument("--type", dest="plan_type", type=int, default=1)
    parser.add_argument("--start", type=str, help="Plan start time, e.g. 2026-05-07T00:00:00")
    parser.add_argument("--end", type=str, help="Plan end time, e.g. 2026-05-10T00:00:00")
    parser.add_argument("--output", type=str, default="outputs/score_radar.png")
    parser.add_argument("--title", type=str, default="Yard Plan Score Radar")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    score = _load_score_json(args.score_json) if args.score_json else _run_plan_score(args)
    output = plot_score_radar(score, args.output, title=args.title, dpi=args.dpi)
    print(f"Radar chart saved to: {output}")


if __name__ == "__main__":
    main()
