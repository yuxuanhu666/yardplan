from __future__ import annotations

import colorsys
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from yardplan_core.models import (
    AllocationGroup,
    BayColumnAllocation,
    ContainerSize,
    PlanningResult,
    _DATA_DIR,
    logger,
)


def _distinct_hex_colors(count: int) -> List[str]:
    """
    Produce `count` fill colors spaced around the hue wheel (golden-ratio steps)
    with staggered saturation/value so neighboring indices stay distinguishable.

    Avoids modulo reuse of a short fixed palette when many allocation groups exist.
    """
    if count <= 0:
        return []
    golden = 0.618033988749895
    out: List[str] = []
    for i in range(count):
        h = (i * golden) % 1.0
        s = 0.58 + 0.32 * ((i % 7) / 6.0)
        v = 0.78 + 0.18 * (((i // 7) % 4) / 3.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


def _plan_parent_group_display_id(
    plan_group_id: str,
    groups_by_id: Dict[str, AllocationGroup],
) -> str:
    """规划项上的 `group_id` 若为子组则显示其父组 id，否则显示自身 id。"""
    group = groups_by_id.get(plan_group_id)
    if group is not None and group.parent_group_id:
        return str(group.parent_group_id)
    return str(plan_group_id)


def _shorten_plan_label(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1] + "…"


@dataclass
class YardVisualizationConfig:
    """Configuration for static 2D yard rendering."""

    stack_width: float = 1.0
    bay_height: float = 0.65
    block_gap_x: float = 4.0
    block_gap_y: float = 3.0
    berth_height: float = 1.2
    margin: float = 1.0
    blocks_per_row: Optional[int] = None
    prefer_real_coordinates: bool = True
    # 使用 nameIndex 真实坐标时若箱区矩形重叠，则自动改用网格铺开
    fallback_if_real_coords_overlap: bool = True
    coordinate_scale: float = 1000.0
    show_bay_labels: bool = True
    show_stack_labels: bool = False
    # 在规划色块中心标注父组 id（无父组则标注当前组 id）
    show_plan_parent_group_label: bool = True
    plan_group_label_fontsize: Optional[float] = None  # None 则按色块尺寸自动估算
    plan_group_label_max_chars: int = 18
    show_legend: bool = True
    existing_color: str = "#9e9e9e"
    empty_edge_color: str = "#d0d0d0"
    block_edge_color: str = "#333333"
    berth_color: str = "#d7ecff"
    conflict_edge_color: str = "#d62728"
    planned_alpha: float = 0.72
    dpi: int = 180
    figsize: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class StackCell:
    block_id: str
    bay_idx: int
    stack_idx: int
    x: float
    y: float
    width: float
    height: float
    occupied: bool


@dataclass
class BlockLayout:
    block_id: str
    x: float
    y: float
    width: float
    height: float
    bay_indices: List[int]
    stack_indices: List[int]
    cells: Dict[Tuple[int, int], StackCell] = field(default_factory=dict)

    def bay_y(self, bay_idx: int) -> float:
        return self.y + self.bay_indices.index(bay_idx) * self.cells_height

    @property
    def cells_height(self) -> float:
        if not self.bay_indices:
            return self.height
        return self.height / len(self.bay_indices)

    @property
    def cells_width(self) -> float:
        if not self.stack_indices:
            return self.width
        return self.width / len(self.stack_indices)


@dataclass
class YardLayout:
    blocks: Dict[str, BlockLayout]
    berth_rect: Tuple[float, float, float, float]
    width: float
    height: float
    used_real_coordinates: bool = False


@dataclass
class PlannedDrawItem:
    group_id: str
    block_id: str
    bay_start: int
    bay_end: int
    stack_start: int
    stack_end: int
    size: Optional[ContainerSize] = None
    is_spanning: bool = False
    is_edge_placement: bool = False


class YardLayoutBuilder:
    """Builds drawing metadata from YardSpace without touching planning logic."""

    def __init__(self, config: Optional[YardVisualizationConfig] = None):
        self.config = config or YardVisualizationConfig()

    def build(
        self,
        yard: Any,
        block_ids: Optional[Sequence[str]] = None,
        name_index_path: Optional[str] = None,
    ) -> YardLayout:
        selected_blocks = self._select_blocks(yard, block_ids)
        block_dims = self._collect_block_dimensions(yard, selected_blocks)
        origins: Dict[str, Tuple[float, float]] = {}
        used_real = False

        if self.config.prefer_real_coordinates:
            origins = self._load_real_block_origins(
                selected_blocks,
                name_index_path or os.path.join(_DATA_DIR, "nameIndex.json"),
            )
            used_real = bool(origins)

        if not origins:
            origins = self._fallback_origins(selected_blocks, block_dims)
        elif len(origins) < len(selected_blocks):
            fallback = self._fallback_origins(selected_blocks, block_dims)
            for block_id in selected_blocks:
                origins.setdefault(block_id, fallback[block_id])

        if (
            used_real
            and self.config.fallback_if_real_coords_overlap
            and len(selected_blocks) > 1
            and self._real_layouts_overlap(selected_blocks, origins, block_dims)
        ):
            logger.info(
                "YardLayoutBuilder: nameIndex 真实坐标下箱区外接矩形重叠，改用网格铺开"
            )
            origins = self._fallback_origins(selected_blocks, block_dims)
            used_real = False

        blocks: Dict[str, BlockLayout] = {}
        min_x = min((origin[0] for origin in origins.values()), default=0.0)
        min_y = min((origin[1] for origin in origins.values()), default=0.0)

        for block_id in selected_blocks:
            bay_indices, stack_indices = block_dims[block_id]
            width = max(1, len(stack_indices)) * self.config.stack_width
            height = max(1, len(bay_indices)) * self.config.bay_height
            raw_x, raw_y = origins[block_id]
            x = raw_x - min_x + self.config.margin
            y = raw_y - min_y + self.config.margin + self.config.berth_height
            layout = BlockLayout(
                block_id=block_id,
                x=x,
                y=y,
                width=width,
                height=height,
                bay_indices=bay_indices,
                stack_indices=stack_indices,
            )
            self._populate_cells(layout, yard)
            blocks[block_id] = layout

        max_x = max((block.x + block.width for block in blocks.values()), default=10.0)
        max_y = max((block.y + block.height for block in blocks.values()), default=10.0)
        berth_rect = (
            self.config.margin,
            self.config.margin * 0.35,
            max_x - self.config.margin,
            self.config.berth_height * 0.7,
        )
        return YardLayout(
            blocks=blocks,
            berth_rect=berth_rect,
            width=max_x + self.config.margin,
            height=max_y + self.config.margin,
            used_real_coordinates=used_real,
        )

    def _select_blocks(self, yard: Any, block_ids: Optional[Sequence[str]]) -> List[str]:
        available = sorted({key[0] for key in yard.stacks.keys()})
        if block_ids is None:
            return available
        wanted = set(block_ids)
        return [block_id for block_id in available if block_id in wanted]

    def _collect_block_dimensions(
        self,
        yard: Any,
        block_ids: Sequence[str],
    ) -> Dict[str, Tuple[List[int], List[int]]]:
        dims: Dict[str, Tuple[set, set]] = {
            block_id: (set(), set()) for block_id in block_ids
        }
        for block_id, bay_idx, stack_idx in yard.stacks.keys():
            if block_id not in dims:
                continue
            dims[block_id][0].add(int(bay_idx))
            dims[block_id][1].add(int(stack_idx))
        return {
            block_id: (sorted(bays), sorted(stacks))
            for block_id, (bays, stacks) in dims.items()
        }

    def _load_real_block_origins(
        self,
        block_ids: Sequence[str],
        name_index_path: str,
    ) -> Dict[str, Tuple[float, float]]:
        if not os.path.exists(name_index_path):
            return {}

        block_set = set(block_ids)
        try:
            with open(name_index_path, "r", encoding="utf-8") as file:
                raw = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load visualization coordinates: %s", exc)
            return {}

        origins: Dict[str, Tuple[float, float]] = {}
        fallback_points: Dict[str, List[Tuple[float, float]]] = {}
        yard_map = raw.get("yardNameIndexMap", {})

        for info in yard_map.values():
            block_id = info.get("blockId")
            if block_id not in block_set:
                continue
            bay_idx = info.get("bayIdx", -1)
            stack_idx = info.get("stackIdx", -1)
            coord = info.get("coordinate") or {}
            if bay_idx is None or stack_idx is None or bay_idx <= 0 or stack_idx <= 0:
                continue
            if "x" not in coord or "y" not in coord:
                continue
            point = (float(coord["x"]) / self.config.coordinate_scale, float(coord["y"]) / self.config.coordinate_scale)
            if info.get("tierIdx", -1) <= 0:
                old = origins.get(block_id)
                origins[block_id] = (
                    min(old[0], point[0]) if old else point[0],
                    min(old[1], point[1]) if old else point[1],
                )
            else:
                fallback_points.setdefault(block_id, []).append(point)

        for block_id, points in fallback_points.items():
            if block_id in origins or not points:
                continue
            origins[block_id] = (min(point[0] for point in points), min(point[1] for point in points))

        return {block_id: origins[block_id] for block_id in block_ids if block_id in origins}

    @staticmethod
    def _aabb_overlap_xywh(
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah

    def _real_layouts_overlap(
        self,
        block_ids: Sequence[str],
        origins: Dict[str, Tuple[float, float]],
        block_dims: Dict[str, Tuple[List[int], List[int]]],
    ) -> bool:
        if len(block_ids) <= 1:
            return False
        present = [bid for bid in block_ids if bid in origins]
        if len(present) <= 1:
            return False
        min_x = min(origins[bid][0] for bid in present)
        min_y = min(origins[bid][1] for bid in present)
        margin = self.config.margin
        berth_y = margin + self.config.berth_height
        rects: List[Tuple[float, float, float, float]] = []
        for bid in present:
            ox, oy = origins[bid]
            x = ox - min_x + margin
            y = oy - min_y + berth_y
            bays, stacks = block_dims[bid]
            w = max(1, len(stacks)) * self.config.stack_width
            h = max(1, len(bays)) * self.config.bay_height
            rects.append((x, y, w, h))
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if self._aabb_overlap_xywh(rects[i], rects[j]):
                    return True
        return False

    def _fallback_origins(
        self,
        block_ids: Sequence[str],
        block_dims: Dict[str, Tuple[List[int], List[int]]],
    ) -> Dict[str, Tuple[float, float]]:
        if not block_ids:
            return {}
        n = len(block_ids)
        if self.config.blocks_per_row:
            per_row = self.config.blocks_per_row
        elif n > 12:
            per_row = max(5, min(8, int(math.ceil(n / 4.0))))
        else:
            per_row = max(1, math.ceil(math.sqrt(n)))
        origins: Dict[str, Tuple[float, float]] = {}
        row_heights: List[float] = []

        for row_start in range(0, len(block_ids), per_row):
            row_blocks = block_ids[row_start : row_start + per_row]
            row_heights.append(
                max(
                    max(1, len(block_dims[block_id][0])) * self.config.bay_height
                    for block_id in row_blocks
                )
            )

        for index, block_id in enumerate(block_ids):
            row = index // per_row
            col = index % per_row
            x = 0.0
            row_blocks = block_ids[row * per_row : row * per_row + col]
            for previous in row_blocks:
                x += max(1, len(block_dims[previous][1])) * self.config.stack_width + self.config.block_gap_x
            y = sum(row_heights[:row]) + row * self.config.block_gap_y
            origins[block_id] = (x, y)
        return origins

    def _populate_cells(self, layout: BlockLayout, yard: Any) -> None:
        for bay_pos, bay_idx in enumerate(layout.bay_indices):
            for stack_pos, stack_idx in enumerate(layout.stack_indices):
                stack_info = yard.stacks.get((layout.block_id, bay_idx, stack_idx), {})
                occupied = _is_stack_occupied(stack_info)
                x = layout.x + stack_pos * self.config.stack_width
                y = layout.y + bay_pos * self.config.bay_height
                layout.cells[(bay_idx, stack_idx)] = StackCell(
                    block_id=layout.block_id,
                    bay_idx=bay_idx,
                    stack_idx=stack_idx,
                    x=x,
                    y=y,
                    width=self.config.stack_width,
                    height=self.config.bay_height,
                    occupied=occupied,
                )


class YardVisualizer:
    """Matplotlib renderer for current yard state and planning overlay."""

    def __init__(self, config: Optional[YardVisualizationConfig] = None):
        self.config = config or YardVisualizationConfig()
        self.layout_builder = YardLayoutBuilder(self.config)

    def plot(
        self,
        yard: Any,
        result: Optional[PlanningResult] = None,
        block_ids: Optional[Sequence[str]] = None,
        save_path: Optional[str] = None,
        name_index_path: Optional[str] = None,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle

        layout = self.layout_builder.build(yard, block_ids=block_ids, name_index_path=name_index_path)
        figsize = self.config.figsize or (
            max(8.0, layout.width * 0.35),
            max(6.0, layout.height * 0.35),
        )
        fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)
        self._draw_base(ax, layout, Rectangle)
        color_map = self._group_colors(result)

        if result is not None:
            self._draw_planning_overlay(ax, layout, result, color_map, Rectangle)

        self._draw_berth(ax, layout, Rectangle)
        self._draw_legend(ax, result, color_map, Patch)
        ax.set_xlim(0, layout.width)
        ax.set_ylim(layout.height, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        subtitle = "real coordinates" if layout.used_real_coordinates else "fallback layout"
        ax.set_title(f"Yard 2D Plan View ({subtitle})")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig, ax

    def _draw_berth(self, ax: Any, layout: YardLayout, rectangle_cls: Any) -> None:
        x, y, width, height = layout.berth_rect
        ax.add_patch(
            rectangle_cls(
                (x, y),
                width,
                height,
                facecolor=self.config.berth_color,
                edgecolor="#4b8bbe",
                linewidth=1.5,
            )
        )
        ax.text(x + width / 2, y + height / 2, "BERTH / QUAY", ha="center", va="center", fontsize=11, weight="bold")

    def _draw_base(self, ax: Any, layout: YardLayout, rectangle_cls: Any) -> None:
        for block in layout.blocks.values():
            ax.add_patch(
                rectangle_cls(
                    (block.x, block.y),
                    block.width,
                    block.height,
                    facecolor="white",
                    edgecolor=self.config.block_edge_color,
                    linewidth=1.3,
                )
            )
            ax.text(block.x, block.y - 0.15, block.block_id, ha="left", va="bottom", fontsize=9, weight="bold")

            for cell in block.cells.values():
                facecolor = self.config.existing_color if cell.occupied else "white"
                ax.add_patch(
                    rectangle_cls(
                        (cell.x, cell.y),
                        cell.width,
                        cell.height,
                        facecolor=facecolor,
                        edgecolor=self.config.empty_edge_color,
                        linewidth=0.35,
                    )
                )

            self._draw_labels(ax, block)

    def _draw_labels(self, ax: Any, block: BlockLayout) -> None:
        if self.config.show_bay_labels:
            for bay_idx in block.bay_indices:
                first = block.cells.get((bay_idx, block.stack_indices[0])) if block.stack_indices else None
                if first:
                    ax.text(first.x - 0.08, first.y + first.height / 2, str(bay_idx), ha="right", va="center", fontsize=5)
        if self.config.show_stack_labels and block.bay_indices:
            top_bay = block.bay_indices[0]
            for stack_idx in block.stack_indices:
                cell = block.cells.get((top_bay, stack_idx))
                if cell:
                    ax.text(cell.x + cell.width / 2, cell.y - 0.05, str(stack_idx), ha="center", va="bottom", fontsize=5)

    def _plan_label_fontsize(self, bbox_w: float, bbox_h: float) -> float:
        """按色块几何尺寸粗略匹配字号（数据坐标）；显式配置优先。"""
        if self.config.plan_group_label_fontsize is not None:
            return float(self.config.plan_group_label_fontsize)
        m = max(1e-6, min(bbox_w, bbox_h))
        return float(max(4.0, min(9.0, m * 12.5)))

    def _draw_plan_parent_center_label(
        self,
        ax: Any,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        plan_group_id: str,
        groups_by_id: Dict[str, AllocationGroup],
        patheffects: Any,
    ) -> None:
        """在规划占位并集矩形中心绘制父组（或顶层组）简写 id。"""
        if not self.config.show_plan_parent_group_label:
            return
        raw = _plan_parent_group_display_id(plan_group_id, groups_by_id)
        label = _shorten_plan_label(raw, self.config.plan_group_label_max_chars)
        if not label:
            return
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        fs = self._plan_label_fontsize(x1 - x0, y1 - y0)
        text = ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            color="#ffffff",
            fontweight="bold",
            clip_on=True,
            zorder=6,
        )
        stroke_w = max(1.15, float(fs * 0.22))
        text.set_path_effects(
            [
                patheffects.withStroke(linewidth=stroke_w, foreground="#232323"),
            ]
        )

    def _draw_planning_overlay(
        self,
        ax: Any,
        layout: YardLayout,
        result: PlanningResult,
        color_map: Dict[str, str],
        rectangle_cls: Any,
    ) -> None:
        from matplotlib import patheffects as mpatheffects

        groups_by_id: Dict[str, AllocationGroup] = {
            g.group_id: g for g in (result.allocation_groups or [])
        }

        for item in extract_planned_draw_items(result, layout):
            block = layout.blocks.get(item.block_id)
            if block is None:
                continue
            color = color_map.get(item.group_id, "#1f77b4")
            cells = _cells_for_item(block, item)
            if not cells:
                continue
            conflict = any(cell.occupied for cell in cells)

            x0 = min(cell.x for cell in cells)
            y0 = min(cell.y for cell in cells)
            x1 = max(cell.x + cell.width for cell in cells)
            y1 = max(cell.y + cell.height for cell in cells)

            if item.is_spanning or item.bay_start != item.bay_end:
                ax.add_patch(
                    rectangle_cls(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        facecolor=color,
                        alpha=self.config.planned_alpha,
                        edgecolor="#004c99" if item.is_edge_placement else "#222222",
                        linewidth=2.2 if item.is_edge_placement else 1.4,
                        hatch="///" if conflict else None,
                    )
                )
                if conflict:
                    ax.add_patch(
                        rectangle_cls(
                            (x0, y0),
                            x1 - x0,
                            y1 - y0,
                            facecolor="none",
                            edgecolor=self.config.conflict_edge_color,
                            linewidth=2.0,
                        )
                    )
                self._draw_plan_parent_center_label(
                    ax, x0, y0, x1, y1, item.group_id, groups_by_id, mpatheffects
                )
                continue

            for cell in cells:
                ax.add_patch(
                    rectangle_cls(
                        (cell.x, cell.y),
                        cell.width,
                        cell.height,
                        facecolor=color,
                        alpha=self.config.planned_alpha,
                        edgecolor=self.config.conflict_edge_color if cell.occupied else "#222222",
                        linewidth=1.5 if cell.occupied else 0.6,
                        hatch="///" if cell.occupied else None,
                    )
                )
            self._draw_plan_parent_center_label(
                ax, x0, y0, x1, y1, item.group_id, groups_by_id, mpatheffects
            )

    def _group_colors(self, result: Optional[PlanningResult]) -> Dict[str, str]:
        if result is None:
            return {}
        group_ids = []
        for item in extract_planned_draw_items(result, None):
            if item.group_id not in group_ids:
                group_ids.append(item.group_id)
        palette = _distinct_hex_colors(len(group_ids))
        return dict(zip(group_ids, palette))

    def _draw_legend(self, ax: Any, result: Optional[PlanningResult], color_map: Dict[str, str], patch_cls: Any) -> None:
        if not self.config.show_legend:
            return
        handles = [
            patch_cls(facecolor=self.config.existing_color, edgecolor="none", label="Existing occupied"),
            patch_cls(facecolor="white", edgecolor=self.config.conflict_edge_color, hatch="///", label="Conflict"),
        ]
        for group_id, color in list(color_map.items())[:20]:
            handles.append(patch_cls(facecolor=color, edgecolor="#222222", label=f"Plan {group_id}"))
        if result is not None and len(color_map) > 20:
            handles.append(patch_cls(facecolor="#ffffff", edgecolor="#222222", label=f"... {len(color_map) - 20} more groups"))
        ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=True)


def build_yard_layout(
    yard: Any,
    block_ids: Optional[Sequence[str]] = None,
    config: Optional[YardVisualizationConfig] = None,
    name_index_path: Optional[str] = None,
) -> YardLayout:
    return YardLayoutBuilder(config).build(yard, block_ids=block_ids, name_index_path=name_index_path)


def plot_yard(
    yard: Any,
    result: Optional[PlanningResult] = None,
    block_ids: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    config: Optional[YardVisualizationConfig] = None,
    name_index_path: Optional[str] = None,
):
    return YardVisualizer(config).plot(
        yard,
        result=result,
        block_ids=block_ids,
        save_path=save_path,
        name_index_path=name_index_path,
    )


def save_yard_visualization(
    yard: Any,
    result: Optional[PlanningResult] = None,
    save_path: str = "yard_plan.png",
    block_ids: Optional[Sequence[str]] = None,
    config: Optional[YardVisualizationConfig] = None,
    name_index_path: Optional[str] = None,
) -> str:
    fig, _ax = plot_yard(
        yard,
        result=result,
        block_ids=block_ids,
        save_path=save_path,
        config=config,
        name_index_path=name_index_path,
    )
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return save_path


def extract_planned_draw_items(
    result: PlanningResult,
    layout: Optional[YardLayout] = None,
) -> List[PlannedDrawItem]:
    if result.bay_column_allocations:
        return _items_from_bay_allocations(result.bay_column_allocations, layout)
    return _items_from_range_plan(result)


def _items_from_bay_allocations(
    allocations: Iterable[BayColumnAllocation],
    layout: Optional[YardLayout],
) -> List[PlannedDrawItem]:
    items: List[PlannedDrawItem] = []
    for allocation in allocations:
        block = layout.blocks.get(allocation.yard_area_id) if layout else None
        if allocation.bay_stack_details:
            for bay_spec, stack_start, stack_end in allocation.bay_stack_details:
                bay_start, bay_end = _bay_range_from_spec(bay_spec)
                items.append(
                    PlannedDrawItem(
                        group_id=allocation.group_id,
                        block_id=allocation.yard_area_id,
                        bay_start=bay_start,
                        bay_end=bay_end,
                        stack_start=int(stack_start),
                        stack_end=int(stack_end),
                        size=allocation.size,
                        is_spanning=allocation.is_spanning,
                        is_edge_placement=allocation.is_edge_placement,
                    )
                )
            continue

        for bay_spec, columns_used in allocation.bay_column_details:
            bay_start, bay_end = _bay_range_from_spec(bay_spec)
            stack_start, stack_end = _stack_range_for_columns(block, columns_used)
            items.append(
                PlannedDrawItem(
                    group_id=allocation.group_id,
                    block_id=allocation.yard_area_id,
                    bay_start=bay_start,
                    bay_end=bay_end,
                    stack_start=stack_start,
                    stack_end=stack_end,
                    size=allocation.size,
                    is_spanning=allocation.is_spanning,
                    is_edge_placement=allocation.is_edge_placement,
                )
            )
    return items


def _items_from_range_plan(result: PlanningResult) -> List[PlannedDrawItem]:
    items: List[PlannedDrawItem] = []
    for row in result.metrics.get("range_plan", {}).get("data", []):
        group_id = str(row.get("groupId", row.get("groupKey", "unknown")))
        for range_item in row.get("rangeList", []):
            items.append(
                PlannedDrawItem(
                    group_id=group_id,
                    block_id=range_item["blockId"],
                    bay_start=int(range_item["startBayIndex"]),
                    bay_end=int(range_item["endBayIndex"]),
                    stack_start=int(range_item.get("startStackIndex", 1)),
                    stack_end=int(range_item.get("endStackIndex", 1)),
                    is_spanning=int(range_item["startBayIndex"]) != int(range_item["endBayIndex"]),
                )
            )
    return items


def _bay_range_from_spec(bay_spec: Any) -> Tuple[int, int]:
    if isinstance(bay_spec, tuple):
        return min(int(bay_spec[0]), int(bay_spec[1])), max(int(bay_spec[0]), int(bay_spec[1]))
    return int(bay_spec), int(bay_spec)


def _stack_range_for_columns(block: Optional[BlockLayout], columns_used: int) -> Tuple[int, int]:
    if block is None or not block.stack_indices:
        return 1, max(1, int(columns_used))
    count = max(1, min(int(columns_used), len(block.stack_indices)))
    selected = block.stack_indices[:count]
    return min(selected), max(selected)


def _cells_for_item(block: BlockLayout, item: PlannedDrawItem) -> List[StackCell]:
    bay_min = min(item.bay_start, item.bay_end)
    bay_max = max(item.bay_start, item.bay_end)
    stack_min = min(item.stack_start, item.stack_end)
    stack_max = max(item.stack_start, item.stack_end)
    cells = []
    for bay_idx in block.bay_indices:
        if not (bay_min <= bay_idx <= bay_max):
            continue
        for stack_idx in block.stack_indices:
            if stack_min <= stack_idx <= stack_max:
                cell = block.cells.get((bay_idx, stack_idx))
                if cell:
                    cells.append(cell)
    return cells


def _is_stack_occupied(stack_info: Dict[str, Any]) -> bool:
    if stack_info.get("top_occupied_tier", 0) > 0:
        return True
    for tier_data in stack_info.get("tiers", {}).values():
        if tier_data.get("occupant") is not None:
            return True
    return False


__all__ = [
    "YardVisualizationConfig",
    "StackCell",
    "BlockLayout",
    "YardLayout",
    "PlannedDrawItem",
    "YardLayoutBuilder",
    "YardVisualizer",
    "build_yard_layout",
    "plot_yard",
    "save_yard_visualization",
    "extract_planned_draw_items",
]
