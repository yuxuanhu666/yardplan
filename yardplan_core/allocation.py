from __future__ import annotations

import math
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from yardplan_core.models import (
    AllocationGroup,
    AreaAssignment,
    Bay,
    BayColumnAllocation,
    BusinessType,
    ContainerSize,
    MAX_TIERS_PER_COLUMN,
    Vessel,
    YardArea,
    logger,
)
from yardplan_core.simultaneous import (
    add_simultaneous_loading_conflict,
    clear_simultaneous_loading_markers,
    group_root_id,
    simultaneous_loading_conflict_group_ids,
)
from yardplan_core.workload import (
    AreaWorkloadProvider,
    AreaWorkloadSnapshot,
    SimulatedAreaWorkloadProvider,
    WorkloadEstimationConfig,
)


@dataclass
class TimeStep:
    """A discrete time bucket used for rolling window planning."""

    step_id: int
    start_time: datetime
    end_time: datetime
    duration_hours: float


@dataclass
class RollingWindow:
    """A rolling planning window containing active groups and capacity forecasts."""

    window_id: str
    step_index: int
    start_time: datetime
    end_time: datetime
    active_groups: List[AllocationGroup] = field(default_factory=list)
    area_capacity_delta: Dict[str, int] = field(default_factory=dict)


class RollingWindowPlanner:
    """Manages rolling window generation and future yard availability updates."""

    def __init__(
        self,
        time_step_hours: float = 4.0,
        window_steps: int = 3,
        flow_events: Optional[List] = None,
    ):
        self.time_step_hours = time_step_hours
        self.window_steps = window_steps
        self.flow_events = flow_events or []

    def generate_time_steps(
        self,
        horizon_start: datetime,
        horizon_end: datetime,
    ) -> List[TimeStep]:
        steps: List[TimeStep] = []
        current = horizon_start
        step_id = 0

        while current < horizon_end:
            end = current + timedelta(hours=self.time_step_hours)
            if end > horizon_end:
                end = horizon_end
            steps.append(
                TimeStep(
                    step_id=step_id,
                    start_time=current,
                    end_time=end,
                    duration_hours=(end - current).total_seconds() / 3600,
                )
            )
            current = end
            step_id += 1

        logger.info(f"Generated {len(steps)} time steps over planning horizon")
        return steps

    def generate_rolling_windows(
        self,
        time_steps: List[TimeStep],
    ) -> List[RollingWindow]:
        windows: List[RollingWindow] = []
        for index in range(len(time_steps) - self.window_steps + 1):
            window_steps = time_steps[index : index + self.window_steps]
            windows.append(
                RollingWindow(
                    window_id=f"WIN-{index:03d}",
                    step_index=index,
                    start_time=window_steps[0].start_time,
                    end_time=window_steps[-1].end_time,
                )
            )

        logger.info(f"Generated {len(windows)} rolling windows")
        return windows

    def update_future_capacity(
        self,
        windows: List[RollingWindow],
        yard_areas: List[YardArea],
        groups: List[AllocationGroup],
    ) -> None:
        # Placeholder for future flow-based workload updates. First-stage scoring
        # consumes area_capacity_delta but current data does not populate flows yet.
        for window in windows:
            window.area_capacity_delta.clear()
            for area in yard_areas:
                window.area_capacity_delta[area.area_id] = 0


class ConstraintChecker:
    """Centralized hard-constraint validation for both stages."""

    @staticmethod
    def can_assign_to_area(group: AllocationGroup, area: YardArea) -> bool:
        if group.business_type != area.business_type:
            return False
        if group.size not in area.supported_sizes:
            return False
        if group.is_edge_only and not area.get_edge_pairs():
            return False
        if (
            group.business_type == BusinessType.EXPORT
            and getattr(area, "_stage1_hard_blocked", False)
        ):
            return False
        return True

    @staticmethod
    def can_place_in_bay(group: AllocationGroup, bay: Bay) -> bool:
        if group.is_large_container_group:
            return bay.can_accept_large()
        return bay.can_accept_20ft()

    @staticmethod
    def validate_45ft_edge(
        placement_bays: List[int],
        yard_area: YardArea,
    ) -> bool:
        if not placement_bays:
            return True
        min_bay = min(placement_bays)
        max_bay = max(placement_bays)
        left_edge = yard_area.bays[0].bay_number if yard_area.bays else 0
        right_edge = yard_area.bays[-1].bay_number if yard_area.bays else 0
        return min_bay == left_edge or max_bay == right_edge


@dataclass
class PlacementPreview:
    """Preview result for first-stage physical feasibility simulation."""

    feasible: bool
    demand: int
    used_20ft_bays: List[Tuple[int, int]] = field(default_factory=list)
    used_large_pairs: List[Tuple[Tuple[int, int], int, bool]] = field(default_factory=list)
    newly_locked_20_bays: Set[int] = field(default_factory=set)
    newly_locked_large_bays: Set[int] = field(default_factory=set)
    fragmentation_cost: float = 0.0
    scarcity_cost: float = 0.0
    reason: str = ""

    @property
    def physical_cost(self) -> float:
        return self.fragmentation_cost + self.scarcity_cost


@dataclass
class AreaResourceState:
    """
    Lightweight first-stage resource model for one YardArea.

    It tracks free tier positions by physical bay number and simulates the
    shared resource conflict between 20ft bay usage and 40/45ft pair usage.
    The tier inventory is sourced from the same Stage2 slot records whenever
    they are available, so Stage1 cannot reserve tiers Stage2 cannot place.
    """

    area_id: str
    bay_free_tiers: Dict[int, int]
    pair_bays: List[Tuple[int, int, bool]]
    locked_20_bays: Set[int]
    locked_large_bays: Set[int]
    initial_total_tiers: int
    initial_large_tiers: int
    initial_edge_tiers: int

    @classmethod
    def from_area(cls, area: YardArea) -> "AreaResourceState":
        single_free_tiers = cls._single_free_tiers(area)
        large_free_tiers = cls._large_free_tiers(area)
        bay_free_tiers: Dict[int, int] = {}
        locked_20_bays = set(getattr(area, "_stage2_existing_20ft_bays", set()))
        locked_large_bays = set(getattr(area, "_stage2_existing_large_bays", set()))
        for bay in area.bays:
            if bay.can_accept_20ft():
                bay_free_tiers[bay.bay_number] = single_free_tiers.get(
                    bay.bay_number,
                    max(0, bay.free_columns) * MAX_TIERS_PER_COLUMN,
                )

        pair_bays: List[Tuple[int, int, bool]] = []
        for pair in area.large_bay_pairs:
            a = pair.bay_a.bay_number
            b = pair.bay_b.bay_number
            pair_free = large_free_tiers.get(
                tuple(sorted((a, b))),
                max(0, pair.free_columns) * MAX_TIERS_PER_COLUMN,
            )
            bay_free_tiers[a] = min(bay_free_tiers.get(a, pair_free), pair_free)
            bay_free_tiers[b] = min(bay_free_tiers.get(b, pair_free), pair_free)
            pair_bays.append((a, b, pair.is_edge_pair))

        initial_total = sum(
            free_tiers
            for bay_number, free_tiers in bay_free_tiers.items()
            if bay_number not in locked_large_bays
        )
        initial_large = sum(
            min(bay_free_tiers.get(a, 0), bay_free_tiers.get(b, 0))
            for a, b, _ in pair_bays
            if cls._large_pair_available(a, b, locked_20_bays, locked_large_bays)
        )
        initial_edge = sum(
            min(bay_free_tiers.get(a, 0), bay_free_tiers.get(b, 0))
            for a, b, is_edge in pair_bays
            if is_edge and cls._large_pair_available(a, b, locked_20_bays, locked_large_bays)
        )
        return cls(
            area_id=area.area_id,
            bay_free_tiers=bay_free_tiers,
            pair_bays=pair_bays,
            locked_20_bays=locked_20_bays,
            locked_large_bays=locked_large_bays,
            initial_total_tiers=max(1, initial_total),
            initial_large_tiers=max(1, initial_large),
            initial_edge_tiers=max(1, initial_edge),
        )

    @staticmethod
    def _single_free_tiers(area: YardArea) -> Dict[int, int]:
        slots = getattr(area, "_stage2_single_slots", None)
        if slots is None:
            return {
                bay.bay_number: max(0, bay.free_columns) * MAX_TIERS_PER_COLUMN
                for bay in area.bays
            }

        capacity: Dict[int, int] = defaultdict(int)
        for slot in slots:
            bay_number = int(slot["bay_number"])
            tiers = {
                int(tier)
                for tier in (slot.get("tiers") or [])
                if 1 <= int(tier) <= MAX_TIERS_PER_COLUMN
            }
            capacity[bay_number] += len(tiers)
        return dict(capacity)

    @staticmethod
    def _large_free_tiers(area: YardArea) -> Dict[Tuple[int, int], int]:
        slots = getattr(area, "_stage2_large_slots", None)
        if slots is None:
            return {
                tuple(sorted((pair.bay_a.bay_number, pair.bay_b.bay_number))): (
                    max(0, pair.free_columns) * MAX_TIERS_PER_COLUMN
                )
                for pair in area.large_bay_pairs
            }

        capacity: Dict[Tuple[int, int], int] = defaultdict(int)
        for slot in slots:
            display_bays = tuple(
                sorted(int(value) for value in slot.get("display_bays", ()))
            )
            if len(display_bays) != 2:
                continue
            tiers = {
                int(tier)
                for tier in (slot.get("tiers") or [])
                if 1 <= int(tier) <= MAX_TIERS_PER_COLUMN
            }
            capacity[display_bays] += len(tiers)
        return dict(capacity)

    @staticmethod
    def _group_tier_demand(group: AllocationGroup) -> int:
        container_count = max(
            0,
            int(group.container_count or len(group.containers) or 0),
        )
        if container_count > 0:
            return container_count
        return max(0, int(group.column_demand or 0)) * MAX_TIERS_PER_COLUMN

    def clone(self) -> "AreaResourceState":
        return AreaResourceState(
            area_id=self.area_id,
            bay_free_tiers=dict(self.bay_free_tiers),
            pair_bays=list(self.pair_bays),
            locked_20_bays=set(self.locked_20_bays),
            locked_large_bays=set(self.locked_large_bays),
            initial_total_tiers=self.initial_total_tiers,
            initial_large_tiers=self.initial_large_tiers,
            initial_edge_tiers=self.initial_edge_tiers,
        )

    def preview_place(
        self,
        group: AllocationGroup,
        demand: Optional[int] = None,
    ) -> PlacementPreview:
        required = demand if demand is not None else self._group_tier_demand(group)
        if required <= 0:
            return PlacementPreview(True, 0)

        if group.size == ContainerSize.SIZE_20:
            return self._preview_20ft(required)
        if group.size == ContainerSize.SIZE_45:
            return self._preview_large(required, edge_only=True)
        return self._preview_large(required, edge_only=False)

    def apply_preview(self, preview: PlacementPreview) -> None:
        if not preview.feasible:
            raise ValueError(f"Cannot apply infeasible preview: {preview.reason}")

        for bay_number, used in preview.used_20ft_bays:
            self.bay_free_tiers[bay_number] = max(0, self.bay_free_tiers.get(bay_number, 0) - used)
        self.locked_20_bays.update(preview.newly_locked_20_bays)

        for (bay_a, bay_b), used, _is_edge in preview.used_large_pairs:
            self.bay_free_tiers[bay_a] = max(0, self.bay_free_tiers.get(bay_a, 0) - used)
            self.bay_free_tiers[bay_b] = max(0, self.bay_free_tiers.get(bay_b, 0) - used)
        self.locked_large_bays.update(preview.newly_locked_large_bays)

    def max_placeable_tiers(self, group: AllocationGroup) -> int:
        if group.size == ContainerSize.SIZE_20:
            return sum(
                free_tiers
                for bay_number, free_tiers in self.bay_free_tiers.items()
                if self._can_use_bay_for_20ft(bay_number)
            )
        if group.size == ContainerSize.SIZE_45:
            return self._max_large_placeable_tiers(edge_only=True)
        return self._max_large_placeable_tiers(edge_only=False)

    def _preview_20ft(self, demand: int) -> PlacementPreview:
        remaining = demand
        used: List[Tuple[int, int]] = []
        candidates = sorted(
            (
                (bay_number, free_tiers)
                for bay_number, free_tiers in self.bay_free_tiers.items()
                if self._can_use_bay_for_20ft(bay_number)
            ),
            key=lambda item: (-item[1], item[0]),
        )

        for bay_number, free_tiers in candidates:
            if remaining <= 0:
                break
            if free_tiers <= 0:
                continue
            take = min(free_tiers, remaining)
            used.append((bay_number, take))
            remaining -= take

        if remaining > 0:
            return PlacementPreview(False, demand, reason="insufficient 20ft tier positions")

        leftover_small = sum(
            1
            for bay_number, used_columns in used
            if 0 < self.bay_free_tiers.get(bay_number, 0) - used_columns <= 1
        )
        fragmentation = (len(used) - 1) * 1.5 + leftover_small * 0.8
        large_after = self._remaining_large_tiers_after_20ft(used)
        large_loss_ratio = max(0.0, 1.0 - large_after / self.initial_large_tiers)
        scarcity = large_loss_ratio * 2.0

        return PlacementPreview(
            feasible=True,
            demand=demand,
            used_20ft_bays=used,
            newly_locked_20_bays={
                bay_number
                for bay_number, _used_columns in used
                if bay_number not in self.locked_20_bays
            },
            fragmentation_cost=fragmentation,
            scarcity_cost=scarcity,
        )

    def _preview_large(self, demand: int, edge_only: bool) -> PlacementPreview:
        remaining = demand
        used: List[Tuple[Tuple[int, int], int, bool]] = []
        simulated_locked_large_bays = set(self.locked_large_bays)

        candidates: List[Tuple[int, int, bool, int]] = []
        for bay_a, bay_b, is_edge in self.pair_bays:
            if edge_only and not is_edge:
                continue
            if not self._large_pair_available(
                bay_a,
                bay_b,
                self.locked_20_bays,
                simulated_locked_large_bays,
            ):
                continue
            free_tiers = min(self.bay_free_tiers.get(bay_a, 0), self.bay_free_tiers.get(bay_b, 0))
            if free_tiers > 0:
                candidates.append((bay_a, bay_b, is_edge, free_tiers))

        candidates.sort(key=lambda item: (item[2], -item[3], item[0], item[1]))

        for bay_a, bay_b, is_edge, free_tiers in candidates:
            if remaining <= 0:
                break
            if not self._large_pair_available(
                bay_a,
                bay_b,
                self.locked_20_bays,
                simulated_locked_large_bays,
            ):
                continue
            take = min(free_tiers, remaining)
            used.append(((bay_a, bay_b), take, is_edge))
            remaining -= take
            simulated_locked_large_bays.update((bay_a, bay_b))

        if remaining > 0:
            resource_name = "edge large-bay tier positions" if edge_only else "large-bay tier positions"
            return PlacementPreview(False, demand, reason=f"insufficient {resource_name}")

        leftover_small = 0
        for (bay_a, bay_b), used_columns, _is_edge in used:
            left = min(self.bay_free_tiers.get(bay_a, 0), self.bay_free_tiers.get(bay_b, 0)) - used_columns
            if 0 < left <= 1:
                leftover_small += 1

        edge_used = sum(columns for _pair, columns, is_edge in used if is_edge)
        total_used = sum(columns for _pair, columns, _is_edge in used)
        fragmentation = (len(used) - 1) * 2.0 + leftover_small
        scarcity = total_used / self.initial_large_tiers
        if edge_only:
            scarcity += total_used / self.initial_edge_tiers * 2.0
        elif edge_used:
            scarcity += edge_used / self.initial_edge_tiers * 1.5

        return PlacementPreview(
            feasible=True,
            demand=demand,
            used_large_pairs=used,
            newly_locked_large_bays={
                bay_number
                for pair, _used_columns, _is_edge in used
                for bay_number in pair
                if bay_number not in self.locked_large_bays
            },
            fragmentation_cost=fragmentation,
            scarcity_cost=scarcity,
        )

    def _remaining_large_tiers_after_20ft(self, used_20ft: List[Tuple[int, int]]) -> int:
        simulated = dict(self.bay_free_tiers)
        for bay_number, used_columns in used_20ft:
            simulated[bay_number] = max(0, simulated.get(bay_number, 0) - used_columns)
        simulated_locked_20_bays = set(self.locked_20_bays)
        simulated_locked_20_bays.update(bay_number for bay_number, _used in used_20ft)
        return self._max_large_tiers_for_locks(
            edge_only=False,
            bay_free_tiers=simulated,
            locked_20_bays=simulated_locked_20_bays,
            locked_large_bays=self.locked_large_bays,
        )

    def _can_use_bay_for_20ft(self, bay_number: int) -> bool:
        return bay_number not in self.locked_large_bays

    def _can_use_pair_for_large(self, bay_a: int, bay_b: int) -> bool:
        return self._large_pair_available(
            bay_a,
            bay_b,
            self.locked_20_bays,
            self.locked_large_bays,
        )

    def _max_large_placeable_tiers(self, edge_only: bool) -> int:
        return self._max_large_tiers_for_locks(
            edge_only=edge_only,
            bay_free_tiers=self.bay_free_tiers,
            locked_20_bays=self.locked_20_bays,
            locked_large_bays=self.locked_large_bays,
        )

    def _max_large_tiers_for_locks(
        self,
        edge_only: bool,
        bay_free_tiers: Dict[int, int],
        locked_20_bays: Set[int],
        locked_large_bays: Set[int],
    ) -> int:
        simulated_locked_large_bays = set(locked_large_bays)
        candidates: List[Tuple[int, int, bool, int]] = []
        for bay_a, bay_b, is_edge in self.pair_bays:
            if edge_only and not is_edge:
                continue
            if not self._large_pair_available(
                bay_a,
                bay_b,
                locked_20_bays,
                simulated_locked_large_bays,
            ):
                continue
            free_tiers = min(bay_free_tiers.get(bay_a, 0), bay_free_tiers.get(bay_b, 0))
            if free_tiers > 0:
                candidates.append((bay_a, bay_b, is_edge, free_tiers))

        total = 0
        candidates.sort(key=lambda item: (item[2], -item[3], item[0], item[1]))
        for bay_a, bay_b, _is_edge, free_tiers in candidates:
            if not self._large_pair_available(
                bay_a,
                bay_b,
                locked_20_bays,
                simulated_locked_large_bays,
            ):
                continue
            total += free_tiers
            simulated_locked_large_bays.update((bay_a, bay_b))
        return total

    @staticmethod
    def _large_pair_available(
        bay_a: int,
        bay_b: int,
        locked_20_bays: Set[int],
        locked_large_bays: Set[int],
    ) -> bool:
        pair = {bay_a, bay_b}
        return not (pair & locked_20_bays or pair & locked_large_bays)


@dataclass
class Stage1LNSConfig:
    """Tunable parameters actually used by stage-1 LNS."""

    max_iterations: int = 120
    no_improve_limit: int = 20
    destroy_fraction: float = 0.2
    stage1_destroy_operator: str = "random"
    stage1_repair_operator: str = "random"
    random_seed: int = 17
    time_limit_seconds: Optional[float] = None
    workload_provider: Optional[str] = None
    workload_soft_capacity_ratio: float = 0.85
    line_small_fragment_weight: float = 5.0
    physical_weight: float = 10.0
    berth_distance_weight: float = 1.0
    busy_profile_weight: float = 160.0
    unassigned_weight: float = 1000.0
    vessel_area_buffer: int = 1
    vessel_area_shortage_weight: float = 20000.0
    vessel_area_excess_weight: float = 2000.0
    dispersion_weight: float = 0.0
    dispersion_share_theta: float = 0.25
    dispersion_min_group_size: int = 100
    tiny_fragment_weight: float = 0.0
    minimum_effective_batch: int = 15
    line_max_area_share_threshold: float = 0.45
    line_share_penalty_weight: float = 24.0
    line_share_penalty_power: float = 2.0
    line_min_total_for_share_penalty: int = 8
    use_simulated_annealing: bool = True
    sa_initial_temperature: float = 50.0
    sa_cooling_rate: float = 0.995
    sa_min_temperature: float = 0.01
    repair_top_k: int = 5
    repair_random_tie_break: bool = True
    repair_tie_tolerance: float = 1e-6
    repair_unassigned_per_iteration: int = 4
    repair_physical_bias_weight: float = 1.0
    repair_congestion_weight: float = 8.0
    repair_split_bias_weight: float = 3.0
    repair_congestion_load_ratio: float = 0.5
    repair_regret_recompute_interval: int = 1
    guard_competition_weight: float = 12.0
    guard_fragment_weight: float = 6.0
    guard_headroom_weight: float = 15.0
    guard_split_weight: float = 3.0
    guard_max_competition_ratio: float = 2.5
    guard_min_headroom_ratio: float = 0.5
    guard_large_group_column_threshold: int = 4
    guard_regret_recompute_interval: int = 1
    simultaneous_loading_pair_count: int = 3
    simultaneous_loading_safety_gap_bays: int = 4
    simultaneous_loading_same_area_weight: float = 2500.0
    alns_reaction_factor: float = 0.2
    alns_segment_length: int = 8
    alns_reward_global_best: float = 8.0
    alns_reward_improving: float = 4.0
    alns_reward_accepted: float = 1.5
    alns_reward_rejected: float = 0.0
    alns_min_operator_weight: float = 0.2
    alns_warmup_rounds: int = 1


@dataclass(init=False)
class Stage1Placement:
    group_id: str
    area_id: str
    container_count: int

    def __init__(
        self,
        group_id: str,
        area_id: str,
        container_count: Optional[int] = None,
        *,
        column_demand: Optional[int] = None,
    ) -> None:
        if container_count is None:
            container_count = column_demand
        elif column_demand is not None and int(column_demand) != int(container_count):
            raise ValueError(
                "Stage1Placement received conflicting container_count and column_demand"
            )
        if container_count is None:
            raise TypeError("Stage1Placement requires container_count")
        self.group_id = group_id
        self.area_id = area_id
        self.container_count = int(container_count)

    @property
    def column_demand(self) -> int:
        """Legacy read adapter; Stage1 stores and uses container_count internally."""
        return self.container_count


@dataclass(init=False)
class Stage1Solution:
    allocations_by_group: Dict[str, Dict[str, int]]
    unassigned_group_ids: set

    def __init__(
        self,
        placements_by_group: Optional[Dict[str, List[Stage1Placement]]] = None,
        unassigned_group_ids: Optional[Set[str]] = None,
        *,
        allocations_by_group: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        self.allocations_by_group = {}
        self.unassigned_group_ids = set(unassigned_group_ids or set())

        for group_id, area_counts in (allocations_by_group or {}).items():
            self.set_group_allocations(group_id, area_counts)
        for group_id, placements in (placements_by_group or {}).items():
            self.set_group_placements(group_id, placements)

    @property
    def placements_by_group(self) -> Dict[str, List[Stage1Placement]]:
        """Compatibility view for callers that still consume placement lists."""
        return {
            group_id: self.group_placements(group_id)
            for group_id in self.allocations_by_group
        }

    @placements_by_group.setter
    def placements_by_group(
        self,
        value: Dict[str, List[Stage1Placement]],
    ) -> None:
        self.allocations_by_group = {}
        for group_id, placements in (value or {}).items():
            self.set_group_placements(group_id, placements)

    def group_placements(self, group_id: str) -> List[Stage1Placement]:
        return [
            Stage1Placement(group_id, area_id, container_count)
            for area_id, container_count in self.allocations_by_group.get(
                group_id,
                {},
            ).items()
        ]

    def iter_group_placements(self):
        for group_id in self.allocations_by_group:
            yield group_id, self.group_placements(group_id)

    def set_group_allocations(
        self,
        group_id: str,
        area_counts: Dict[str, int],
    ) -> None:
        self.allocations_by_group.pop(group_id, None)
        for area_id, container_count in area_counts.items():
            self.add_allocation(group_id, area_id, container_count)

    def set_group_placements(
        self,
        group_id: str,
        placements: List[Stage1Placement],
    ) -> None:
        self.allocations_by_group.pop(group_id, None)
        for placement in placements:
            self.add_allocation(
                group_id,
                placement.area_id,
                placement.container_count,
            )

    def add_allocation(
        self,
        group_id: str,
        area_id: str,
        container_count: int,
    ) -> None:
        count = int(container_count or 0)
        if count < 0:
            raise ValueError("Stage1 allocation container_count cannot be negative")
        if count == 0:
            return
        area_counts = self.allocations_by_group.setdefault(group_id, {})
        area_counts[area_id] = area_counts.get(area_id, 0) + count

    def remove_group(self, group_id: str) -> None:
        self.allocations_by_group.pop(group_id, None)

    def allocated_container_count(self, group_id: str) -> int:
        return sum(self.allocations_by_group.get(group_id, {}).values())

    def validate_conservation(
        self,
        expected_container_counts: Dict[str, int],
    ) -> None:
        unknown_group_ids = (
            set(self.allocations_by_group) | set(self.unassigned_group_ids)
        ) - set(expected_container_counts)
        if unknown_group_ids:
            raise ValueError(
                f"Stage1 solution contains unknown groups: {sorted(unknown_group_ids)}"
            )

        for group_id, expected_count in expected_container_counts.items():
            expected = max(0, int(expected_count or 0))
            allocated = self.allocated_container_count(group_id)
            unassigned = expected if group_id in self.unassigned_group_ids else 0
            if allocated + unassigned != expected:
                raise ValueError(
                    "Stage1 container conservation failed: "
                    f"group={group_id} allocated={allocated} "
                    f"unassigned={unassigned} expected={expected}"
                )

    def clone(self) -> "Stage1Solution":
        return Stage1Solution(
            allocations_by_group={
                group_id: dict(area_counts)
                for group_id, area_counts in self.allocations_by_group.items()
            },
            unassigned_group_ids=set(self.unassigned_group_ids),
        )


@dataclass
class Stage1CostContext:
    states: Dict[str, AreaResourceState]
    area_load: Dict[str, int]
    line_area_load: Dict[Tuple[Optional[int], str], int]
    physical_cost: float = 0.0
    berth_distance_cost: float = 0.0
    busy_profile_cost: float = 0.0
    infeasible_count: int = 0

    def clone(self) -> "Stage1CostContext":
        return Stage1CostContext(
            states={area_id: state.clone() for area_id, state in self.states.items()},
            area_load=defaultdict(int, self.area_load),
            line_area_load=defaultdict(int, self.line_area_load),
            physical_cost=self.physical_cost,
            berth_distance_cost=self.berth_distance_cost,
            busy_profile_cost=self.busy_profile_cost,
            infeasible_count=self.infeasible_count,
        )


@dataclass(frozen=True)
class Stage1CostBreakdown:
    unassigned_cost: float = 0.0
    infeasible_cost: float = 0.0
    physical_cost: float = 0.0
    berth_distance_cost: float = 0.0
    busy_profile_cost: float = 0.0
    vessel_area_cost: float = 0.0
    dispersion_cost: float = 0.0
    tiny_fragment_cost: float = 0.0
    simultaneous_loading_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class Stage1DerivedStats:
    vessel_area_load: Dict[Tuple[str, BusinessType, str], int] = field(
        default_factory=dict
    )
    vessel_active_areas: Dict[Tuple[str, BusinessType], Set[str]] = field(
        default_factory=dict
    )
    group_area_load: Dict[Tuple[str, str], int] = field(default_factory=dict)
    vessel_export_total: Dict[str, int] = field(default_factory=dict)
    group_export_share: Dict[str, float] = field(default_factory=dict)
    group_compatible_area_count: Dict[str, int] = field(default_factory=dict)
    vessel_actual_area_count: Dict[Tuple[str, BusinessType], int] = field(
        default_factory=dict
    )
    vessel_preferred_area_count: Dict[Tuple[str, BusinessType], int] = field(
        default_factory=dict
    )


@dataclass
class Stage1AssignmentCandidate:
    delta_cost: float
    placements: List[Stage1Placement]
    previews: List[Tuple[str, PlacementPreview]]


@dataclass
class ALNSOperatorState:
    weight: float = 1.0
    segment_score: float = 0.0
    segment_uses: int = 0
    total_score: float = 0.0
    total_uses: int = 0


class AreaScoringStrategy(ABC):
    @abstractmethod
    def score(
        self,
        group: AllocationGroup,
        area: YardArea,
        current_assignments: Dict,
    ) -> float:
        raise NotImplementedError


class DefaultAreaScoringStrategy(AreaScoringStrategy):
    """Compatibility scorer retained for callers that inject a simple strategy."""

    def score(
        self,
        group: AllocationGroup,
        area: YardArea,
        current_assignments: Dict,
    ) -> float:
        if not ConstraintChecker.can_assign_to_area(group, area):
            return -float("inf")
        projected = (
            current_assignments.get(area.area_id, 0)
            + AreaResourceState._group_tier_demand(group)
        )
        return -projected


class Stage1YardAreaAssigner:
    """
    Stage 1: assign allocation groups to one or more yard areas.

    The implementation uses a constraint-aware initial solution followed by a
    small LNS loop with optional simulated annealing. It only produces
    AreaAssignment and unassigned groups; exact
    bay/range allocation remains a second-stage concern.

    Time-step workload is currently ignored in the first-stage objective.
    Yard capacity checks use tier positions, matching Stage2 placement units.
    """

    def __init__(
        self,
        scoring_strategy: Optional[AreaScoringStrategy] = None,
        config: Optional[Stage1LNSConfig] = None,
        workload_provider: Optional[AreaWorkloadProvider] = None,
        workload_estimation_config: Optional[WorkloadEstimationConfig] = None,
    ):
        self.scorer = scoring_strategy or DefaultAreaScoringStrategy()
        self.config = config or Stage1LNSConfig()
        self._rng = random.Random(self.config.random_seed)
        self.workload_provider: Optional[AreaWorkloadProvider] = (
            workload_provider
            if workload_provider is not None
            else self._resolve_workload_provider()
        )
        self.workload_estimation_config = (
            workload_estimation_config or WorkloadEstimationConfig()
        )
        self.workload_snapshot: Optional[AreaWorkloadSnapshot] = None
        self._vessels: Dict[str, Vessel] = {}
        self._area_by_id: Dict[str, YardArea] = {}
        self._time_steps: List[TimeStep] = []
        self._voyage_step_span: Dict[str, List[int]] = {}
        self._last_search_summary: Dict[str, object] = {}
        self._last_convergence_history: List[Dict[str, Any]] = []
        self._last_cost_breakdown: Optional[Stage1CostBreakdown] = None
        self._last_derived_stats: Optional[Stage1DerivedStats] = None

    def _resolve_workload_provider(self) -> Optional[AreaWorkloadProvider]:
        provider_name = (self.config.workload_provider or "").lower()
        if not provider_name:
            return None
        if provider_name == "simulated":
            return SimulatedAreaWorkloadProvider()
        raise ValueError(f"未知 workload_provider: {self.config.workload_provider!r}")

    def set_workload_snapshot(
        self,
        snapshot: Optional[AreaWorkloadSnapshot],
    ) -> None:
        self.workload_snapshot = snapshot
        if snapshot is not None:
            self._time_steps = list(snapshot.steps)

    def build_workload_snapshot(
        self,
        *,
        yard_areas: List[YardArea],
        time_steps: List[TimeStep],
        vessels: Dict[str, Vessel],
        groups: List[AllocationGroup],
        plan_start_time: datetime,
        plan_end_time: datetime,
    ) -> Optional[AreaWorkloadSnapshot]:
        self._cache_planning_context(vessels=vessels, time_steps=time_steps)
        if not time_steps or self.workload_provider is None:
            self.workload_snapshot = None
            return None
        workload_config = replace(
            self.workload_estimation_config,
            time_step_hours=time_steps[0].duration_hours,
        )
        snapshot = self.workload_provider.build_snapshot(
            yard_areas=yard_areas,
            time_steps=time_steps,
            vessels=vessels,
            groups=groups,
            plan_start_time=plan_start_time,
            plan_end_time=plan_end_time,
            config=workload_config,
        )
        self.workload_snapshot = snapshot
        return snapshot

    def _reset_workload_context(self) -> None:
        self._vessels = {}
        self._time_steps = []
        self._voyage_step_span = {}

    def _cache_planning_context(
        self,
        *,
        vessels: Dict[str, Vessel],
        time_steps: List[TimeStep],
    ) -> None:
        self._vessels = dict(vessels)
        self._time_steps = list(time_steps)
        self._voyage_step_span = {
            voyage_id: self._compute_voyage_step_span(vessel, time_steps)
            for voyage_id, vessel in vessels.items()
        }

    @staticmethod
    def _compute_voyage_step_span(
        vessel: Vessel,
        time_steps: List[TimeStep],
    ) -> List[int]:
        eta = vessel.eta
        etd = vessel.etd or eta + timedelta(hours=48)
        if etd <= eta:
            etd = eta + timedelta(hours=48)

        span: List[int] = []
        for index, step in enumerate(time_steps):
            is_last = index == len(time_steps) - 1
            overlaps = step.start_time < etd and step.end_time > eta
            if not overlaps and is_last and eta == step.end_time:
                overlaps = True
            if overlaps:
                span.append(step.step_id)
        return span

    def assign(
        self,
        groups: List[AllocationGroup],
        yard_areas: List[YardArea],
    ) -> Tuple[List[AreaAssignment], List[AllocationGroup]]:
        original_groups = list(groups)
        area_by_id = {area.area_id: area for area in yard_areas}
        self._area_by_id = dict(area_by_id)
        self._mark_simultaneous_loading_conflicts(original_groups)

        solution = self._construct_initial_solution(original_groups, yard_areas)
        best = self._run_lns(solution, original_groups, yard_areas)
        assignments, unassigned, generated_groups = self._materialize_solution(
            best,
            original_groups,
            area_by_id,
        )

        existing_ids = {group.group_id for group in groups}
        for generated in generated_groups:
            if generated.group_id not in existing_ids:
                groups.append(generated)
                existing_ids.add(generated.group_id)

        search_label = self._search_algorithm_label()
        logger.info(
            "Stage1 %s complete: %s assignments, %s split groups, %s unassigned",
            search_label,
            len(assignments),
            len(generated_groups),
            len(unassigned),
        )
        return assignments, unassigned

    def _mark_simultaneous_loading_conflicts(
        self,
        groups: List[AllocationGroup],
    ) -> List[Tuple[str, str, str]]:
        for group in groups:
            clear_simultaneous_loading_markers(group)

        pair_count = max(0, int(self.config.simultaneous_loading_pair_count or 0))
        if pair_count <= 0:
            return []

        groups_by_voyage_root: Dict[str, Dict[str, List[AllocationGroup]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for group in groups:
            if group.business_type != BusinessType.EXPORT or not group.voyage_id:
                continue
            groups_by_voyage_root[str(group.voyage_id)][group_root_id(group)].append(group)

        selected_pairs: List[Tuple[str, str, str]] = []
        gap = max(0, int(self.config.simultaneous_loading_safety_gap_bays or 0))
        for voyage_id, groups_by_root in sorted(groups_by_voyage_root.items()):
            root_ids = sorted(groups_by_root)
            if len(root_ids) < 2:
                continue

            rng = random.Random(
                f"{self.config.random_seed}:simultaneous-loading:{voyage_id}"
            )
            rng.shuffle(root_ids)
            for index in range(min(pair_count, len(root_ids) // 2)):
                root_a = root_ids[index * 2]
                root_b = root_ids[index * 2 + 1]
                pair_id = f"{voyage_id}-SL{index + 1}"
                for group in groups_by_root[root_a]:
                    add_simultaneous_loading_conflict(group, root_b, pair_id, gap)
                for group in groups_by_root[root_b]:
                    add_simultaneous_loading_conflict(group, root_a, pair_id, gap)
                selected_pairs.append((voyage_id, root_a, root_b))

        if selected_pairs:
            logger.info("Stage1 simultaneous loading pairs: %s", selected_pairs)
        return selected_pairs

    def _build_states(self, yard_areas: List[YardArea]) -> Dict[str, AreaResourceState]:
        return {area.area_id: AreaResourceState.from_area(area) for area in yard_areas}

    @staticmethod
    def _group_tier_demand(group: AllocationGroup) -> int:
        return AreaResourceState._group_tier_demand(group)

    @staticmethod
    def _tier_demand_to_columns(tier_demand: int) -> int:
        demand = max(0, int(tier_demand or 0))
        if demand <= 0:
            return 0
        return int(math.ceil(demand / MAX_TIERS_PER_COLUMN))

    def _construct_initial_solution(
        self,
        groups: List[AllocationGroup],
        yard_areas: List[YardArea],
    ) -> Stage1Solution:
        solution = Stage1Solution()
        group_by_id = {group.group_id: group for group in groups}
        ordered = sorted(groups, key=lambda group: self._difficulty_key(group, yard_areas))

        for group in ordered:
            candidate = self._try_assign_group(
                solution,
                group,
                group_by_id,
                yard_areas,
            )
            if candidate is None:
                solution.remove_group(group.group_id)
                solution.unassigned_group_ids.add(group.group_id)
                continue
            solution.set_group_placements(group.group_id, candidate.placements)
            solution.unassigned_group_ids.discard(group.group_id)

        return solution

    def _run_lns(
        self,
        initial: Stage1Solution,
        groups: List[AllocationGroup],
        yard_areas: List[YardArea],
    ) -> Stage1Solution:
        group_by_id = {group.group_id: group for group in groups}
        best = initial.clone()
        best_cost = self.evaluate(best, group_by_id, yard_areas)
        current = best.clone()
        current_cost = best_cost
        no_improve = 0
        started_at = time.monotonic()
        temperature = self.config.sa_initial_temperature
        destroy_pool = self._destroy_operator_pool()
        repair_pool = self._repair_operator_pool()
        destroy_states = {
            operator: ALNSOperatorState() for operator in destroy_pool
        }
        repair_states = {
            operator: ALNSOperatorState() for operator in repair_pool
        }
        adaptive_mode = self._use_adaptive_operator_selection()
        iterations_completed = 0
        history: List[Dict[str, Any]] = [
            {
                "iteration": 0,
                "currentCost": round(current_cost, 6),
                "candidateCost": round(current_cost, 6),
                "bestCost": round(best_cost, 6),
                "accepted": True,
                "delta": 0.0,
                "currentUnassigned": len(current.unassigned_group_ids),
                "candidateUnassigned": len(current.unassigned_group_ids),
                "bestUnassigned": len(best.unassigned_group_ids),
                "temperature": round(temperature, 6),
            }
        ]

        for iteration in range(self.config.max_iterations):
            if self.config.time_limit_seconds is not None:
                if time.monotonic() - started_at >= self.config.time_limit_seconds:
                    break
            if no_improve >= self.config.no_improve_limit:
                break

            destroy_operator = self._select_alns_operator(
                iteration=iteration,
                operators=destroy_pool,
                states=destroy_states,
            )
            removed_ids = self._apply_destroy_operator(
                current,
                group_by_id,
                yard_areas,
                operator=destroy_operator,
            )
            if not removed_ids:
                break

            candidate = current.clone()
            for group_id in removed_ids:
                candidate.remove_group(group_id)
                candidate.unassigned_group_ids.discard(group_id)

            repair_ids = self._repair_ids_with_unassigned(
                candidate,
                removed_ids,
                group_by_id,
                yard_areas,
            )

            repair_operator = self._select_alns_operator(
                iteration=iteration,
                operators=repair_pool,
                states=repair_states,
            )
            repaired = self._apply_repair_operator(
                candidate,
                repair_ids,
                group_by_id,
                yard_areas,
                operator=repair_operator,
            )
            candidate_cost = self.evaluate(repaired, group_by_id, yard_areas)

            delta = candidate_cost - current_cost
            accept = delta < 0
            if (
                not accept
                and self.config.use_simulated_annealing
                and temperature >= self.config.sa_min_temperature
            ):
                accept = self._rng.random() < math.exp(-delta / temperature)

            reward = self._alns_iteration_reward(
                candidate_cost=candidate_cost,
                current_cost=current_cost,
                best_cost=best_cost,
                accepted=accept,
            )
            self._record_alns_feedback(destroy_states[destroy_operator], reward)
            self._record_alns_feedback(repair_states[repair_operator], reward)

            if accept:
                current = repaired
                current_cost = candidate_cost
                if candidate_cost < best_cost:
                    best = repaired.clone()
                    best_cost = candidate_cost
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if self.config.use_simulated_annealing:
                temperature = max(
                    self.config.sa_min_temperature,
                    temperature * self.config.sa_cooling_rate,
                )

            iterations_completed = iteration + 1
            history.append(
                {
                    "iteration": iterations_completed,
                    "currentCost": round(current_cost, 6),
                    "candidateCost": round(candidate_cost, 6),
                    "bestCost": round(best_cost, 6),
                    "accepted": bool(accept),
                    "delta": round(delta, 6),
                    "currentUnassigned": len(current.unassigned_group_ids),
                    "candidateUnassigned": len(repaired.unassigned_group_ids),
                    "bestUnassigned": len(best.unassigned_group_ids),
                    "temperature": round(temperature, 6),
                }
            )
            if adaptive_mode:
                self._maybe_refresh_alns_weights(
                    iteration=iterations_completed,
                    states=destroy_states,
                )
                self._maybe_refresh_alns_weights(
                    iteration=iterations_completed,
                    states=repair_states,
                )

        self._log_unassigned_diagnostics(best, group_by_id, yard_areas)
        self._last_search_summary = {
            "search_label": self._search_algorithm_label(),
            "iterations": iterations_completed,
            "adaptive_mode": adaptive_mode,
            "destroy": self._operator_summary(destroy_states),
            "repair": self._operator_summary(repair_states),
        }
        self._log_alns_summary()
        self._last_derived_stats = self.derive_solution_stats(
            best,
            group_by_id,
            yard_areas,
        )
        self._last_cost_breakdown = self.evaluate_cost_breakdown(
            best,
            group_by_id,
            yard_areas,
        )
        self._log_cost_breakdown(self._last_cost_breakdown)
        if self.config.use_simulated_annealing:
            logger.info(
                "Stage1 %s best cost: %.3f, unassigned=%s (SA final T=%.4f)",
                self._search_algorithm_label(),
                best_cost,
                len(best.unassigned_group_ids),
                temperature,
            )
        else:
            logger.info(
                "Stage1 %s best cost: %.3f, unassigned=%s",
                self._search_algorithm_label(),
                best_cost,
                len(best.unassigned_group_ids),
            )
        self._last_convergence_history = history
        return best

    def _repair_ids_with_unassigned(
        self,
        partial: Stage1Solution,
        removed_ids: List[str],
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
    ) -> List[str]:
        limit = max(0, int(self.config.repair_unassigned_per_iteration or 0))
        unassigned_ids = [
            group_id
            for group_id in partial.unassigned_group_ids
            if group_id in group_by_id
        ]
        unassigned_ids.sort(
            key=lambda group_id: self._unassigned_repair_priority(
                group_by_id[group_id],
                yard_areas,
            )
        )
        if limit > 0:
            unassigned_ids = unassigned_ids[:limit]

        ordered: List[str] = []
        seen: Set[str] = set()
        for group_id in unassigned_ids + list(removed_ids):
            if group_id in seen or group_id not in group_by_id:
                continue
            ordered.append(group_id)
            seen.add(group_id)
        return ordered

    def _unassigned_repair_priority(
        self,
        group: AllocationGroup,
        yard_areas: List[YardArea],
    ) -> Tuple[int, int, int, int]:
        return self._difficulty_key(group, yard_areas)

    def _use_adaptive_operator_selection(self) -> bool:
        return (
            len(self._destroy_operator_pool()) > 1
            or len(self._repair_operator_pool()) > 1
        )

    def _search_algorithm_label(self) -> str:
        return "ALNS" if self._use_adaptive_operator_selection() else "LNS"

    def _destroy_operator_pool(self) -> List[str]:
        destroy = (self.config.stage1_destroy_operator or "random").lower()
        if destroy in {"adaptive", "auto", "random"}:
            return ["random"]
        raise ValueError(f"未知 stage1_destroy_operator: {destroy!r}")

    def _repair_operator_pool(self) -> List[str]:
        repair = (self.config.stage1_repair_operator or "random").lower()
        if repair in {"adaptive", "auto", "random"}:
            return ["random"]
        raise ValueError(f"未知 stage1_repair_operator: {repair!r}")

    def _select_alns_operator(
        self,
        *,
        iteration: int,
        operators: List[str],
        states: Dict[str, ALNSOperatorState],
    ) -> str:
        if len(operators) == 1:
            return operators[0]
        warmup_span = max(0, int(self.config.alns_warmup_rounds)) * len(operators)
        if iteration < warmup_span:
            return operators[iteration % len(operators)]

        total_weight = sum(max(self.config.alns_min_operator_weight, states[op].weight) for op in operators)
        draw = self._rng.random() * total_weight
        cumulative = 0.0
        for operator in operators:
            cumulative += max(self.config.alns_min_operator_weight, states[operator].weight)
            if draw <= cumulative:
                return operator
        return operators[-1]

    def _alns_iteration_reward(
        self,
        *,
        candidate_cost: float,
        current_cost: float,
        best_cost: float,
        accepted: bool,
    ) -> float:
        if candidate_cost < best_cost:
            return self.config.alns_reward_global_best
        if candidate_cost < current_cost:
            return self.config.alns_reward_improving
        if accepted:
            return self.config.alns_reward_accepted
        return self.config.alns_reward_rejected

    @staticmethod
    def _record_alns_feedback(state: ALNSOperatorState, reward: float) -> None:
        state.segment_score += reward
        state.segment_uses += 1
        state.total_score += reward
        state.total_uses += 1

    def _maybe_refresh_alns_weights(
        self,
        *,
        iteration: int,
        states: Dict[str, ALNSOperatorState],
    ) -> None:
        segment_length = max(1, int(self.config.alns_segment_length))
        if iteration % segment_length != 0:
            return
        reaction = min(1.0, max(0.0, self.config.alns_reaction_factor))
        min_weight = max(0.0, self.config.alns_min_operator_weight)
        for state in states.values():
            if state.segment_uses > 0:
                average_reward = state.segment_score / state.segment_uses
                state.weight = max(
                    min_weight,
                    (1.0 - reaction) * state.weight + reaction * average_reward,
                )
            else:
                state.weight = max(min_weight, state.weight)
            state.segment_score = 0.0
            state.segment_uses = 0

    @staticmethod
    def _operator_summary(
        states: Dict[str, ALNSOperatorState],
    ) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for operator, state in states.items():
            avg_reward = state.total_score / state.total_uses if state.total_uses else 0.0
            summary[operator] = {
                "weight": round(state.weight, 4),
                "uses": float(state.total_uses),
                "avg_reward": round(avg_reward, 4),
            }
        return summary

    def _log_alns_summary(self) -> None:
        if not self._last_search_summary:
            return
        logger.info(
            "Stage1 %s operator summary: iterations=%s destroy=%s repair=%s",
            self._last_search_summary.get("search_label", "LNS"),
            self._last_search_summary.get("iterations", 0),
            self._last_search_summary.get("destroy", {}),
            self._last_search_summary.get("repair", {}),
        )

    @staticmethod
    def _log_cost_breakdown(breakdown: Stage1CostBreakdown) -> None:
        logger.info(
            "Stage1 cost breakdown: total=%.3f unassigned=%.3f "
            "infeasible=%.3f physical=%.3f berth_distance=%.3f "
            "busy_profile=%.3f vessel_area=%.3f dispersion=%.3f "
            "tiny_fragment=%.3f simultaneous_loading=%.3f",
            breakdown.total_cost,
            breakdown.unassigned_cost,
            breakdown.infeasible_cost,
            breakdown.physical_cost,
            breakdown.berth_distance_cost,
            breakdown.busy_profile_cost,
            breakdown.vessel_area_cost,
            breakdown.dispersion_cost,
            breakdown.tiny_fragment_cost,
            breakdown.simultaneous_loading_cost,
        )

    def evaluate(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        *,
        enforce_vessel_area_min: bool = True,
    ) -> float:
        return self.evaluate_cost_breakdown(
            solution,
            group_by_id,
            yard_areas,
            enforce_vessel_area_min=enforce_vessel_area_min,
        ).total_cost

    def derive_solution_stats(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
    ) -> Stage1DerivedStats:
        vessel_area_load: Dict[Tuple[str, BusinessType, str], int] = defaultdict(int)
        vessel_active_areas: Dict[Tuple[str, BusinessType], Set[str]] = defaultdict(set)
        group_area_load: Dict[Tuple[str, str], int] = {}
        vessel_export_total: Dict[str, int] = defaultdict(int)
        group_export_share: Dict[str, float] = {
            group_id: 0.0 for group_id in group_by_id
        }
        group_compatible_area_count: Dict[str, int] = {}
        compatible_export_areas_by_voyage: Dict[str, Set[str]] = defaultdict(set)

        for group_id, group in group_by_id.items():
            compatible_area_ids = {
                area.area_id
                for area in yard_areas
                if ConstraintChecker.can_assign_to_area(group, area)
            }
            group_compatible_area_count[group_id] = len(compatible_area_ids)
            if group.business_type == BusinessType.EXPORT and group.voyage_id:
                vessel_export_total[str(group.voyage_id)] += self._group_tier_demand(
                    group
                )
                compatible_export_areas_by_voyage[str(group.voyage_id)].update(
                    compatible_area_ids
                )

        for group_id, group in group_by_id.items():
            if group.business_type != BusinessType.EXPORT or not group.voyage_id:
                continue
            total = vessel_export_total.get(str(group.voyage_id), 0)
            if total > 0:
                group_export_share[group_id] = self._group_tier_demand(group) / total

        for group_id, area_counts in solution.allocations_by_group.items():
            group = group_by_id.get(group_id)
            if group is None or not group.voyage_id:
                continue
            vessel_key = (str(group.voyage_id), group.business_type)
            for area_id, raw_count in area_counts.items():
                container_count = max(0, int(raw_count or 0))
                if container_count <= 0:
                    continue
                group_area_load[(group_id, area_id)] = container_count
                vessel_area_load[(*vessel_key, area_id)] += container_count
                vessel_active_areas[vessel_key].add(area_id)

        vessel_actual_area_count = {
            vessel_key: len(area_ids)
            for vessel_key, area_ids in vessel_active_areas.items()
        }
        vessel_preferred_area_count: Dict[Tuple[str, BusinessType], int] = {}
        buffer_areas = max(0, int(self.config.vessel_area_buffer or 0))
        for voyage_id, export_total in vessel_export_total.items():
            if export_total <= 0:
                continue
            vessel = self._vessels.get(voyage_id)
            if vessel is None:
                continue
            try:
                eqp_num = int(getattr(vessel, "eqp_num", 0) or 0)
            except (TypeError, ValueError):
                continue
            if eqp_num <= 0:
                continue
            vessel_key = (voyage_id, BusinessType.EXPORT)
            compatible_area_count = len(
                compatible_export_areas_by_voyage.get(voyage_id, set())
            )
            vessel_preferred_area_count[vessel_key] = min(
                compatible_area_count,
                eqp_num + buffer_areas,
            )
            vessel_actual_area_count.setdefault(vessel_key, 0)

        return Stage1DerivedStats(
            vessel_area_load=dict(vessel_area_load),
            vessel_active_areas={
                vessel_key: set(area_ids)
                for vessel_key, area_ids in vessel_active_areas.items()
            },
            group_area_load=group_area_load,
            vessel_export_total=dict(vessel_export_total),
            group_export_share=group_export_share,
            group_compatible_area_count=group_compatible_area_count,
            vessel_actual_area_count=vessel_actual_area_count,
            vessel_preferred_area_count=vessel_preferred_area_count,
        )

    def evaluate_cost_breakdown(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        *,
        enforce_vessel_area_min: bool = True,
    ) -> Stage1CostBreakdown:
        context = self._replay_solution(solution, group_by_id, yard_areas)
        derived_stats = self.derive_solution_stats(
            solution,
            group_by_id,
            yard_areas,
        )
        del enforce_vessel_area_min
        unassigned_container_count = self._unassigned_container_count(
            solution,
            group_by_id,
        )

        unassigned_cost = (
            unassigned_container_count * self.config.unassigned_weight
        )
        infeasible_cost = context.infeasible_count * self.config.unassigned_weight
        physical_cost = context.physical_cost * self.config.physical_weight
        berth_distance_cost = (
            context.berth_distance_cost * self.config.berth_distance_weight
        )
        busy_profile_cost = (
            context.busy_profile_cost * self.config.busy_profile_weight
        )
        vessel_area_cost = self._vessel_area_cost(derived_stats)
        dispersion_cost = self._dispersion_cost(
            solution,
            group_by_id,
            derived_stats,
        )
        tiny_fragment_cost = self._tiny_fragment_cost(solution, group_by_id)
        simultaneous_loading_cost = self._simultaneous_loading_same_area_cost(
            solution,
            group_by_id,
        )

        total_cost = infeasible_cost + unassigned_cost
        total_cost += physical_cost
        total_cost += berth_distance_cost
        total_cost += busy_profile_cost
        total_cost += simultaneous_loading_cost
        total_cost += vessel_area_cost
        total_cost += dispersion_cost
        total_cost += tiny_fragment_cost

        return Stage1CostBreakdown(
            unassigned_cost=unassigned_cost,
            infeasible_cost=infeasible_cost,
            physical_cost=physical_cost,
            berth_distance_cost=berth_distance_cost,
            busy_profile_cost=busy_profile_cost,
            vessel_area_cost=vessel_area_cost,
            dispersion_cost=dispersion_cost,
            tiny_fragment_cost=tiny_fragment_cost,
            simultaneous_loading_cost=simultaneous_loading_cost,
            total_cost=total_cost,
        )

    def _vessel_area_cost(
        self,
        stats: Stage1DerivedStats,
    ) -> float:
        shortage_weight = max(
            0.0,
            float(self.config.vessel_area_shortage_weight or 0.0),
        )
        excess_weight = max(
            0.0,
            float(self.config.vessel_area_excess_weight or 0.0),
        )
        cost = 0.0
        for vessel_key, preferred_count in stats.vessel_preferred_area_count.items():
            actual_count = stats.vessel_actual_area_count.get(vessel_key, 0)
            shortage = max(0, preferred_count - actual_count)
            excess = max(0, actual_count - preferred_count)
            cost += shortage_weight * shortage * shortage
            cost += excess_weight * excess * excess
        return cost

    def _dispersion_cost(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
        stats: Stage1DerivedStats,
    ) -> float:
        weight = max(0.0, float(self.config.dispersion_weight or 0.0))
        if weight <= 0.0:
            return 0.0

        cost = 0.0
        for group_id, area_counts in solution.allocations_by_group.items():
            group = group_by_id.get(group_id)
            if group is None or group.business_type != BusinessType.EXPORT:
                continue
            target = self._target_group_dispersion(group, stats)
            effective = self._effective_area_count(area_counts)
            deficit = max(0.0, target - effective)
            cost += weight * deficit * deficit
        return cost

    def _tiny_fragment_cost(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
    ) -> float:
        del group_by_id
        weight = max(0.0, float(self.config.tiny_fragment_weight or 0.0))
        minimum_batch = max(1, int(self.config.minimum_effective_batch or 1))
        if weight <= 0.0:
            return 0.0

        cost = 0.0
        for area_counts in solution.allocations_by_group.values():
            for raw_count in area_counts.values():
                container_count = max(0, int(raw_count or 0))
                if 0 < container_count < minimum_batch:
                    shortage_ratio = (minimum_batch - container_count) / minimum_batch
                    cost += weight * shortage_ratio * shortage_ratio
        return cost

    @staticmethod
    def _effective_area_count(area_counts: Dict[str, int]) -> float:
        positive_counts = [
            max(0, int(container_count or 0))
            for container_count in area_counts.values()
            if int(container_count or 0) > 0
        ]
        total = sum(positive_counts)
        if total <= 0:
            return 0.0
        concentration = sum((container_count / total) ** 2 for container_count in positive_counts)
        if concentration <= 0.0:
            return 0.0
        return 1.0 / concentration

    def _target_group_dispersion(
        self,
        group: AllocationGroup,
        stats: Stage1DerivedStats,
    ) -> float:
        compatible_count = max(
            0,
            int(stats.group_compatible_area_count.get(group.group_id, 0) or 0),
        )
        if compatible_count <= 0:
            return 0.0
        if group.business_type != BusinessType.EXPORT or not group.voyage_id:
            return 1.0

        vessel_key = (str(group.voyage_id), BusinessType.EXPORT)
        preferred_count = max(
            0,
            int(stats.vessel_preferred_area_count.get(vessel_key, 0) or 0),
        )
        target_cap = min(preferred_count, compatible_count)
        if target_cap <= 0:
            return 0.0

        group_size = self._group_tier_demand(group)
        minimum_group_size = max(
            0,
            int(self.config.dispersion_min_group_size or 0),
        )
        if group_size < minimum_group_size:
            return 1.0

        share = max(0.0, float(stats.group_export_share.get(group.group_id, 0.0)))
        theta = max(1e-6, float(self.config.dispersion_share_theta or 0.0))
        raw_target = 1.0 + (preferred_count - 1.0) * (
            1.0 - math.exp(-share / theta)
        )
        return max(1.0, min(raw_target, float(target_cap)))

    def _unassigned_container_count(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
    ) -> int:
        total = 0
        for group_id in solution.unassigned_group_ids:
            group = group_by_id.get(group_id)
            if group is None:
                total += 1
                continue
            total += self._group_tier_demand(group)
        return total

    def _simultaneous_loading_same_area_cost(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
    ) -> float:
        weight = float(self.config.simultaneous_loading_same_area_weight or 0.0)
        if weight <= 0.0:
            return 0.0

        areas_by_root: Dict[str, Set[str]] = defaultdict(set)
        roots_by_voyage: Dict[str, Set[str]] = defaultdict(set)
        root_groups: Dict[str, AllocationGroup] = {}
        for group_id, placements in solution.iter_group_placements():
            group = group_by_id.get(group_id)
            if group is None:
                continue
            root_id = group_root_id(group)
            root_groups.setdefault(root_id, group)
            if group.voyage_id:
                roots_by_voyage[str(group.voyage_id)].add(root_id)
            for placement in placements:
                if placement.area_id:
                    areas_by_root[root_id].add(placement.area_id)

        pair_keys: Set[Tuple[str, str, str]] = set()
        for voyage_id, root_ids in roots_by_voyage.items():
            for root_id in root_ids:
                group = root_groups.get(root_id)
                if group is None:
                    continue
                for other_id in simultaneous_loading_conflict_group_ids(group):
                    if other_id not in root_ids:
                        continue
                    left, right = sorted((root_id, other_id))
                    pair_keys.add((voyage_id, left, right))

        conflict_count = 0
        for _voyage_id, root_a, root_b in pair_keys:
            shared_areas = areas_by_root.get(root_a, set()) & areas_by_root.get(root_b, set())
            conflict_count += len(shared_areas)
        return weight * conflict_count

    def evaluate_placement_delta(
        self,
        partial: Stage1Solution,
        group: AllocationGroup,
        placements: List[Stage1Placement],
        previews: List[Tuple[str, PlacementPreview]],
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        base_cost: Optional[float] = None,
    ) -> float:
        del previews
        before_cost = (
            base_cost
            if base_cost is not None
            else self.evaluate(partial, group_by_id, yard_areas)
        )
        candidate = partial.clone()
        candidate.set_group_placements(group.group_id, placements)
        candidate.unassigned_group_ids.discard(group.group_id)
        after_cost = self.evaluate(candidate, group_by_id, yard_areas)
        return after_cost - before_cost

    def _random_destroy(self, solution: Stage1Solution) -> List[str]:
        assigned_ids = list(solution.allocations_by_group)
        if not assigned_ids:
            return []
        remove_count = max(1, int(math.ceil(len(assigned_ids) * self.config.destroy_fraction)))
        return self._rng.sample(assigned_ids, min(remove_count, len(assigned_ids)))

    def _apply_destroy_operator(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        operator: Optional[str] = None,
    ) -> List[str]:
        del group_by_id, yard_areas
        op = (operator or self.config.stage1_destroy_operator or "random").lower()
        if op in {"adaptive", "auto", "random"}:
            return self._random_destroy(solution)
        raise ValueError(f"未知 stage1_destroy_operator: {op!r}")

    def _area_fragmentation_score(self, state: AreaResourceState) -> float:
        partial_20ft_bays = sum(
            1
            for bay_number in state.locked_20_bays
            if state.bay_free_tiers.get(bay_number, 0) > 0
        )
        partial_large_pairs = sum(
            1
            for bay_a, bay_b, _is_edge in state.pair_bays
            if bay_a in state.locked_large_bays
            and bay_b in state.locked_large_bays
            and min(state.bay_free_tiers.get(bay_a, 0), state.bay_free_tiers.get(bay_b, 0)) > 0
        )
        return float(partial_20ft_bays + partial_large_pairs)

    def _resolve_repair_operator(self) -> str:
        repair = (self.config.stage1_repair_operator or "auto").lower()
        if repair in {"adaptive", "auto", "random"}:
            return "random"
        raise ValueError(f"未知 stage1_repair_operator: {self.config.stage1_repair_operator!r}")

    def _apply_repair_operator(
        self,
        partial: Stage1Solution,
        group_ids: List[str],
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        operator: Optional[str] = None,
    ) -> Stage1Solution:
        op = (operator or self._resolve_repair_operator()).lower()
        if op in {"adaptive", "auto", "random"}:
            return self._random_repair(
                partial,
                group_ids,
                group_by_id,
                yard_areas,
            )
        raise ValueError(f"未知 stage1_repair_operator: {op!r}")

    def _random_repair(
        self,
        partial: Stage1Solution,
        group_ids: List[str],
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
    ) -> Stage1Solution:
        ordered = [
            group_by_id[group_id]
            for group_id in group_ids
            if group_id in group_by_id
        ]
        self._rng.shuffle(ordered)
        ordered.sort(
            key=lambda group: (
                0 if group.group_id in partial.unassigned_group_ids else 1,
                self._unassigned_repair_priority(group, yard_areas),
            )
        )
        for group in ordered:
            candidate = self._try_assign_group(partial, group, group_by_id, yard_areas)
            if candidate is None:
                partial.remove_group(group.group_id)
                partial.unassigned_group_ids.add(group.group_id)
                continue
            partial.set_group_placements(group.group_id, candidate.placements)
            partial.unassigned_group_ids.discard(group.group_id)
        return partial

    def _workload_balanced_repair(
        self,
        partial: Stage1Solution,
        group_ids: List[str],
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
    ) -> Stage1Solution:
        if not group_ids:
            return partial
        if self.workload_snapshot is None:
            return self._random_repair(partial, group_ids, group_by_id, yard_areas)

        pending = [group_by_id[group_id] for group_id in group_ids if group_id in group_by_id]
        if not pending:
            return partial

        regret_by_group: Dict[str, float] = {}
        recompute_interval = max(1, int(self.config.repair_regret_recompute_interval))
        processed_since_recompute = recompute_interval

        while pending:
            if processed_since_recompute >= recompute_interval or not regret_by_group:
                regret_by_group = {
                    group.group_id: self._estimate_repair_regret(
                        partial,
                        group,
                        group_by_id,
                        yard_areas,
                    )
                    for group in pending
                }
                processed_since_recompute = 0

            selected_group = max(
                pending,
                key=lambda group: (
                    regret_by_group.get(group.group_id, 0.0),
                    self._rng.random(),
                ),
            )
            regret = regret_by_group.get(selected_group.group_id, float("inf"))
            context = self._replay_solution(partial, group_by_id, yard_areas)
            base_cost = self.evaluate(partial, group_by_id, yard_areas)
            candidate = self._try_assign_group(
                partial,
                selected_group,
                group_by_id,
                yard_areas,
                repair_mode="workload_balanced",
            )
            if candidate is None:
                partial.remove_group(selected_group.group_id)
                partial.unassigned_group_ids.add(selected_group.group_id)
            else:
                best_tie_score = self._workload_balanced_tie_score(
                    candidate,
                    selected_group,
                    partial,
                    context,
                    group_by_id,
                    yard_areas,
                    base_cost,
                )
                partial.set_group_placements(
                    selected_group.group_id,
                    candidate.placements,
                )
                partial.unassigned_group_ids.discard(selected_group.group_id)
                logger.debug(
                    "Stage1 balanced repair: group=%s regret=%.3f tie_score=%.3f",
                    selected_group.group_id,
                    regret,
                    best_tie_score,
                )

            pending = [
                group for group in pending if group.group_id != selected_group.group_id
            ]
            regret_by_group.pop(selected_group.group_id, None)
            processed_since_recompute += 1

        return partial

    def _stage2_guarded_repair(
        self,
        partial: Stage1Solution,
        group_ids: List[str],
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
    ) -> Stage1Solution:
        if not group_ids:
            return partial
        if self.workload_snapshot is None:
            return self._random_repair(partial, group_ids, group_by_id, yard_areas)

        pending = [group_by_id[group_id] for group_id in group_ids if group_id in group_by_id]
        if not pending:
            return partial

        regret_by_group: Dict[str, float] = {}
        recompute_interval = max(1, int(self.config.guard_regret_recompute_interval))
        processed_since_recompute = recompute_interval

        while pending:
            if processed_since_recompute >= recompute_interval or not regret_by_group:
                regret_by_group = {
                    group.group_id: self._estimate_repair_regret(
                        partial,
                        group,
                        group_by_id,
                        yard_areas,
                        repair_mode="stage2_guarded",
                    )
                    for group in pending
                }
                processed_since_recompute = 0

            selected_group = max(
                pending,
                key=lambda group: (
                    regret_by_group.get(group.group_id, 0.0),
                    self._repair_group_difficulty(group),
                    self._rng.random(),
                ),
            )
            regret = regret_by_group.get(selected_group.group_id, float("inf"))
            context = self._replay_solution(partial, group_by_id, yard_areas)
            base_cost = self.evaluate(partial, group_by_id, yard_areas)
            candidate = self._try_assign_group(
                partial,
                selected_group,
                group_by_id,
                yard_areas,
                repair_mode="stage2_guarded",
            )
            if candidate is None:
                partial.remove_group(selected_group.group_id)
                partial.unassigned_group_ids.add(selected_group.group_id)
            else:
                best_tie_score = self._stage2_guarded_tie_score(
                    candidate,
                    selected_group,
                    partial,
                    context,
                    group_by_id,
                    yard_areas,
                    base_cost,
                )
                partial.set_group_placements(
                    selected_group.group_id,
                    candidate.placements,
                )
                partial.unassigned_group_ids.discard(selected_group.group_id)
                logger.debug(
                    "Stage1 guarded repair: group=%s regret=%.3f tie_score=%.3f",
                    selected_group.group_id,
                    regret,
                    best_tie_score,
                )

            pending = [
                group for group in pending if group.group_id != selected_group.group_id
            ]
            regret_by_group.pop(selected_group.group_id, None)
            processed_since_recompute += 1

        return partial

    def _generate_repair_candidates(
        self,
        partial: Stage1Solution,
        group: AllocationGroup,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        context: Stage1CostContext,
        base_cost: float,
    ) -> List[Stage1AssignmentCandidate]:
        demand = self._group_tier_demand(group)
        if demand <= 0:
            return []

        stats = self.derive_solution_stats(partial, group_by_id, yard_areas)
        vessel_key = (str(group.voyage_id), group.business_type)
        active_area_ids = stats.vessel_active_areas.get(vessel_key, set())
        vessel_load_by_area = {
            area_id: stats.vessel_area_load.get((*vessel_key, area_id), 0)
            for area_id in active_area_ids
        }

        area_capacity: Dict[str, int] = {}
        for area in yard_areas:
            if not ConstraintChecker.can_assign_to_area(group, area):
                continue
            capacity = min(
                context.states[area.area_id].max_placeable_tiers(group),
                self._planning_remaining_capacity(
                    area.area_id,
                    context.area_load,
                    context.states,
                ),
            )
            if capacity > 0:
                area_capacity[area.area_id] = capacity

        if sum(area_capacity.values()) < demand:
            return []

        existing_ids = sorted(
            (area_id for area_id in area_capacity if area_id in active_area_ids),
            key=lambda area_id: (
                vessel_load_by_area.get(area_id, 0),
                -area_capacity[area_id],
                area_id,
            ),
        )
        new_ids = sorted(
            (area_id for area_id in area_capacity if area_id not in active_area_ids),
            key=lambda area_id: (-area_capacity[area_id], area_id),
        )
        capacity_order = sorted(
            area_capacity,
            key=lambda area_id: (-area_capacity[area_id], area_id),
        )
        existing_first_order = existing_ids + new_ids

        target = self._target_group_dispersion(group, stats)
        minimum_required = self._minimum_area_count_for_capacity(
            demand,
            capacity_order,
            area_capacity,
        )
        target_counts = {
            minimum_required,
            max(1, int(math.floor(target))),
            max(1, int(math.ceil(target))),
        }
        target_counts = {
            min(len(area_capacity), demand, count)
            for count in target_counts
            if count > 0
        }

        distributions: List[Dict[str, int]] = []
        seen_shapes: Set[Tuple[Tuple[str, int], ...]] = set()

        def add_distribution(distribution: Optional[Dict[str, int]]) -> None:
            if not distribution:
                return
            normalized = {
                area_id: int(container_count)
                for area_id, container_count in distribution.items()
                if int(container_count) > 0
            }
            if sum(normalized.values()) != demand:
                return
            shape = tuple(sorted(normalized.items()))
            if shape in seen_shapes:
                return
            seen_shapes.add(shape)
            distributions.append(normalized)

        for area_id, capacity in area_capacity.items():
            if capacity >= demand:
                add_distribution({area_id: demand})

        for order in (existing_first_order, capacity_order):
            add_distribution(
                self._greedy_capacity_distribution(demand, order, area_capacity)
            )
            for area_count in sorted(target_counts):
                selected = order[:area_count]
                add_distribution(
                    self._balanced_capacity_distribution(
                        demand,
                        selected,
                        area_capacity,
                    )
                )

        for new_area_id in new_ids:
            for area_count in sorted(target_counts):
                if area_count <= 1:
                    selected = [new_area_id]
                else:
                    selected = existing_ids[: area_count - 1] + [new_area_id]
                if len(selected) != area_count:
                    continue
                add_distribution(
                    self._balanced_capacity_distribution(
                        demand,
                        selected,
                        area_capacity,
                    )
                )

        for excluded_area_id in existing_ids:
            close_order = [
                area_id
                for area_id in existing_first_order
                if area_id != excluded_area_id
            ]
            add_distribution(
                self._greedy_capacity_distribution(
                    demand,
                    close_order,
                    area_capacity,
                )
            )

        for distribution in list(distributions):
            add_distribution(
                self._merge_tiny_distribution(distribution, area_capacity)
            )

        candidates = [
            candidate
            for distribution in distributions
            for candidate in [
                self._candidate_from_distribution(
                    partial,
                    group,
                    group_by_id,
                    yard_areas,
                    context,
                    base_cost,
                    distribution,
                )
            ]
            if candidate is not None
        ]
        candidates.sort(key=self._candidate_sort_key)
        return candidates

    @staticmethod
    def _minimum_area_count_for_capacity(
        demand: int,
        area_order: List[str],
        area_capacity: Dict[str, int],
    ) -> int:
        accumulated = 0
        for index, area_id in enumerate(area_order, start=1):
            accumulated += max(0, int(area_capacity.get(area_id, 0)))
            if accumulated >= demand:
                return index
        return len(area_order) + 1

    @staticmethod
    def _greedy_capacity_distribution(
        demand: int,
        area_order: List[str],
        area_capacity: Dict[str, int],
    ) -> Optional[Dict[str, int]]:
        remaining = max(0, int(demand))
        distribution: Dict[str, int] = {}
        for area_id in area_order:
            if remaining <= 0:
                break
            take = min(remaining, max(0, int(area_capacity.get(area_id, 0))))
            if take <= 0:
                continue
            distribution[area_id] = take
            remaining -= take
        return distribution if remaining == 0 else None

    @staticmethod
    def _balanced_capacity_distribution(
        demand: int,
        area_ids: List[str],
        area_capacity: Dict[str, int],
    ) -> Optional[Dict[str, int]]:
        selected = [
            area_id
            for area_id in area_ids
            if max(0, int(area_capacity.get(area_id, 0))) > 0
        ]
        remaining = max(0, int(demand))
        if remaining <= 0 or not selected or remaining < len(selected):
            return None
        if sum(area_capacity[area_id] for area_id in selected) < remaining:
            return None

        distribution: Dict[str, int] = {}
        for index, area_id in enumerate(selected):
            remaining_ids = selected[index + 1 :]
            remaining_slots = len(remaining_ids)
            future_capacity = sum(area_capacity[item] for item in remaining_ids)
            minimum_take = max(1, remaining - future_capacity)
            maximum_take = min(
                area_capacity[area_id],
                remaining - remaining_slots,
            )
            if minimum_take > maximum_take:
                return None
            ideal_take = int(math.ceil(remaining / (remaining_slots + 1)))
            take = max(minimum_take, min(maximum_take, ideal_take))
            distribution[area_id] = take
            remaining -= take
        return distribution if remaining == 0 else None

    def _merge_tiny_distribution(
        self,
        distribution: Dict[str, int],
        area_capacity: Dict[str, int],
    ) -> Dict[str, int]:
        minimum_batch = max(1, int(self.config.minimum_effective_batch or 1))
        merged = dict(distribution)
        tiny_area_ids = sorted(
            (
                area_id
                for area_id, container_count in merged.items()
                if 0 < container_count < minimum_batch
            ),
            key=lambda area_id: (merged[area_id], area_id),
        )
        for area_id in tiny_area_ids:
            if area_id not in merged:
                continue
            fragment = merged[area_id]
            receivers = sorted(
                (
                    other_id
                    for other_id in merged
                    if other_id != area_id
                    and area_capacity.get(other_id, 0) > merged[other_id]
                ),
                key=lambda other_id: (
                    -(area_capacity[other_id] - merged[other_id]),
                    other_id,
                ),
            )
            remaining = fragment
            for receiver_id in receivers:
                spare = area_capacity[receiver_id] - merged[receiver_id]
                move = min(spare, remaining)
                merged[receiver_id] += move
                remaining -= move
                if remaining <= 0:
                    break
            if remaining <= 0:
                del merged[area_id]
            else:
                moved = fragment - remaining
                merged[area_id] -= moved
        return merged

    def _candidate_from_distribution(
        self,
        partial: Stage1Solution,
        group: AllocationGroup,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        context: Stage1CostContext,
        base_cost: float,
        distribution: Dict[str, int],
    ) -> Optional[Stage1AssignmentCandidate]:
        placements: List[Stage1Placement] = []
        previews: List[Tuple[str, PlacementPreview]] = []
        for area_id, container_count in sorted(distribution.items()):
            if container_count <= 0:
                return None
            if container_count > self._planning_remaining_capacity(
                area_id,
                context.area_load,
                context.states,
            ):
                return None
            preview = context.states[area_id].preview_place(
                group,
                demand=container_count,
            )
            if not preview.feasible:
                return None
            placements.append(
                Stage1Placement(group.group_id, area_id, container_count)
            )
            previews.append((area_id, preview))

        delta_cost = self.evaluate_placement_delta(
            partial,
            group,
            placements,
            previews,
            group_by_id,
            yard_areas,
            base_cost=base_cost,
        )
        if not math.isfinite(delta_cost):
            return None
        return Stage1AssignmentCandidate(
            delta_cost=delta_cost,
            placements=placements,
            previews=previews,
        )

    def _estimate_repair_regret(
        self,
        partial: Stage1Solution,
        group: AllocationGroup,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        repair_mode: str = "workload_balanced",
    ) -> float:
        context = self._replay_solution(partial, group_by_id, yard_areas)
        base_cost = self.evaluate(partial, group_by_id, yard_areas)
        candidates = self._generate_repair_candidates(
            partial,
            group,
            group_by_id,
            yard_areas,
            context,
            base_cost,
        )
        if not candidates:
            return float("inf")
        if repair_mode == "stage2_guarded":
            guarded_candidates = self._filter_stage2_guarded_candidates(
                candidates,
                group,
                partial,
                context,
                group_by_id,
                yard_areas,
                base_cost,
            )
            if guarded_candidates:
                candidates = guarded_candidates

        sort_scores = sorted(
            self._repair_tie_score(
                candidate,
                group,
                partial,
                context,
                group_by_id,
                yard_areas,
                base_cost,
                repair_mode,
            )
            for candidate in candidates
        )
        if len(sort_scores) == 1:
            return float("inf")
        return sort_scores[1] - sort_scores[0]

    def _try_assign_group(
        self,
        partial: Stage1Solution,
        group: AllocationGroup,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        repair_mode: str = "random",
    ) -> Optional[Stage1AssignmentCandidate]:
        context = self._replay_solution(partial, group_by_id, yard_areas)
        base_cost = self.evaluate(partial, group_by_id, yard_areas)
        candidates = self._generate_repair_candidates(
            partial,
            group,
            group_by_id,
            yard_areas,
            context,
            base_cost,
        )
        if not candidates:
            return None
        return self._select_repair_candidate(
            candidates,
            group,
            repair_mode=repair_mode,
            partial=partial,
            context=context,
            group_by_id=group_by_id,
            yard_areas=yard_areas,
            base_cost=base_cost,
        )

    def _select_repair_candidate(
        self,
        candidates: List[Stage1AssignmentCandidate],
        group: AllocationGroup,
        *,
        repair_mode: str = "random",
        partial: Optional[Stage1Solution] = None,
        context: Optional[Stage1CostContext] = None,
        group_by_id: Optional[Dict[str, AllocationGroup]] = None,
        yard_areas: Optional[List[YardArea]] = None,
        base_cost: Optional[float] = None,
    ) -> Stage1AssignmentCandidate:
        if repair_mode == "random":
            candidates.sort(key=self._candidate_sort_key)
            if not self.config.repair_random_tie_break:
                return candidates[0]
            best_cost = candidates[0].delta_cost
            top_k = max(1, min(self.config.repair_top_k, len(candidates)))
            tied_candidates = [
                candidate
                for candidate in candidates[:top_k]
                if candidate.delta_cost <= best_cost + self.config.repair_tie_tolerance
            ]
            return self._rng.choice(tied_candidates)

        if repair_mode not in {"workload_balanced", "stage2_guarded"}:
            raise ValueError(f"未知 repair_mode: {repair_mode!r}")
        if (
            partial is None
            or context is None
            or group_by_id is None
            or yard_areas is None
            or base_cost is None
        ):
            raise ValueError(f"{repair_mode} repair 需要完整的 partial/context/group_by_id/yard_areas/base_cost")

        if repair_mode == "stage2_guarded":
            filtered_candidates = self._filter_stage2_guarded_candidates(
                candidates,
                group,
                partial,
                context,
                group_by_id,
                yard_areas,
                base_cost,
            )
            if filtered_candidates:
                if len(filtered_candidates) < len(candidates):
                    logger.debug(
                        "Stage1 guarded repair filtered candidates: group=%s before=%s after=%s",
                        group.group_id,
                        len(candidates),
                        len(filtered_candidates),
                    )
                candidates = filtered_candidates

        candidates.sort(key=self._candidate_sort_key)
        top_k = max(1, min(self.config.repair_top_k, len(candidates)))
        best_cost = candidates[0].delta_cost
        if repair_mode == "stage2_guarded":
            tied_candidates = candidates
        else:
            tied_candidates = [
                candidate
                for candidate in candidates[:top_k]
                if candidate.delta_cost <= best_cost + self.config.repair_tie_tolerance
            ]
        scored_candidates = sorted(
            (
                (
                    self._repair_tie_score(
                        candidate,
                        group,
                        partial,
                        context,
                        group_by_id,
                        yard_areas,
                        base_cost,
                        repair_mode,
                    ),
                    self._candidate_sort_key(candidate),
                    candidate,
                )
                for candidate in tied_candidates
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not self.config.repair_random_tie_break:
            return scored_candidates[0][2]
        best_tie_score = scored_candidates[0][0]
        final_candidates = [
            candidate
            for tie_score, _sort_key, candidate in scored_candidates
            if tie_score <= best_tie_score + self.config.repair_tie_tolerance
        ]
        return self._rng.choice(final_candidates)

    def _candidate_sort_key(
        self,
        candidate: Stage1AssignmentCandidate,
    ) -> Tuple[float, int, Tuple[Tuple[str, int], ...]]:
        return (
            candidate.delta_cost,
            len(candidate.placements),
            tuple(
                (placement.area_id, placement.container_count)
                for placement in candidate.placements
            ),
        )

    def _replay_solution(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
    ) -> Stage1CostContext:
        area_by_id = {area.area_id: area for area in yard_areas}
        states = self._build_states(yard_areas)
        area_load: Dict[str, int] = defaultdict(int)
        line_area_load: Dict[Tuple[Optional[int], str], int] = defaultdict(int)
        physical_cost = 0.0
        berth_distance_cost = 0.0
        busy_profile_cost = 0.0
        infeasible_count = 0

        for group_id, placements in solution.iter_group_placements():
            group = group_by_id.get(group_id)
            if group is None:
                infeasible_count += 1
                continue
            group_infeasible = False
            if (
                sum(placement.container_count for placement in placements)
                != self._group_tier_demand(group)
            ):
                group_infeasible = True
            for placement in placements:
                if placement.container_count <= 0:
                    group_infeasible = True
                    continue
                area = area_by_id.get(placement.area_id)
                if area is None or not ConstraintChecker.can_assign_to_area(group, area):
                    group_infeasible = True
                    continue
                state = states[placement.area_id]
                preview = state.preview_place(group, placement.container_count)
                if not preview.feasible:
                    group_infeasible = True
                    continue
                state.apply_preview(preview)
                physical_cost += preview.physical_cost
                berth_distance_cost += self._placement_berth_distance_cost(
                    group,
                    area,
                    placement.container_count,
                )
                busy_profile_cost += self._placement_busy_profile_cost(
                    group,
                    area,
                    placement.container_count,
                )
                area_load[placement.area_id] += placement.container_count
                line_area_load[(group.line_key, placement.area_id)] += (
                    placement.container_count
                )
            if group_infeasible:
                infeasible_count += 1

        for area_id, load in area_load.items():
            if load > self._planning_capacity(area_id, states):
                infeasible_count += 1

        return Stage1CostContext(
            states=states,
            area_load=area_load,
            line_area_load=line_area_load,
            physical_cost=physical_cost,
            berth_distance_cost=berth_distance_cost,
            busy_profile_cost=busy_profile_cost,
            infeasible_count=infeasible_count,
        )

    def _apply_preview_to_context(
        self,
        context: Stage1CostContext,
        group: AllocationGroup,
        placement: Stage1Placement,
        preview: PlacementPreview,
    ) -> None:
        context.states[placement.area_id].apply_preview(preview)
        context.area_load[placement.area_id] += placement.container_count
        context.line_area_load[(group.line_key, placement.area_id)] += (
            placement.container_count
        )
        context.physical_cost += preview.physical_cost
        area = self._area_by_id.get(placement.area_id)
        context.berth_distance_cost += self._placement_berth_distance_cost(
            group,
            area,
            placement.container_count,
        )
        context.busy_profile_cost += self._placement_busy_profile_cost(
            group,
            area,
            placement.container_count,
        )

    def _area_congestion_stats(
        self,
        partial: Stage1Solution,
        yard_areas: List[YardArea],
        context: Stage1CostContext,
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        group_count_by_area: Dict[str, int] = defaultdict(int)
        load_ratio_by_area: Dict[str, float] = {}
        for _group_id, placements in partial.iter_group_placements():
            for placement in placements:
                group_count_by_area[placement.area_id] += 1
        for area in yard_areas:
            capacity = max(1, self._planning_capacity(area.area_id, context.states))
            load_ratio_by_area[area.area_id] = (
                context.area_load.get(area.area_id, 0) / capacity
            )
        return dict(group_count_by_area), load_ratio_by_area

    def _workload_balanced_tie_score(
        self,
        candidate: Stage1AssignmentCandidate,
        group: AllocationGroup,
        partial: Stage1Solution,
        context: Stage1CostContext,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        base_cost: float,
    ) -> float:
        del group_by_id, base_cost
        temp_context = context.clone()
        physical_bias = 0.0

        for placement, preview_entry in zip(
            candidate.placements,
            candidate.previews,
        ):
            _area_id, preview = preview_entry
            if preview is None:
                continue
            self._apply_preview_to_context(
                temp_context,
                group,
                placement,
                preview,
            )
            physical_bias += preview.physical_cost

        group_count_by_area, load_ratio_by_area = self._area_congestion_stats(
            partial,
            yard_areas,
            context,
        )
        congestion_penalty = 0.0
        for placement in candidate.placements:
            congestion_penalty += group_count_by_area.get(placement.area_id, 0)
            congestion_penalty += (
                self.config.repair_congestion_load_ratio
                * load_ratio_by_area.get(placement.area_id, 0.0)
            )

        split_penalty = max(0, len(candidate.placements) - 1)
        return (
            candidate.delta_cost
            + self.config.repair_physical_bias_weight * physical_bias
            + self.config.repair_congestion_weight * congestion_penalty
            + self.config.repair_split_bias_weight * split_penalty
        )

    def _repair_tie_score(
        self,
        candidate: Stage1AssignmentCandidate,
        group: AllocationGroup,
        partial: Stage1Solution,
        context: Stage1CostContext,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        base_cost: float,
        repair_mode: str,
    ) -> float:
        if repair_mode == "stage2_guarded":
            return self._stage2_guarded_tie_score(
                candidate,
                group,
                partial,
                context,
                group_by_id,
                yard_areas,
                base_cost,
            )
        return self._workload_balanced_tie_score(
            candidate,
            group,
            partial,
            context,
            group_by_id,
            yard_areas,
            base_cost,
        )

    def _stage2_guarded_tie_score(
        self,
        candidate: Stage1AssignmentCandidate,
        group: AllocationGroup,
        partial: Stage1Solution,
        context: Stage1CostContext,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        base_cost: float,
    ) -> float:
        del group_by_id, base_cost
        affected_area_ids = {placement.area_id for placement in candidate.placements}
        before_fragment = {
            area_id: self._area_fragmentation_score(context.states[area_id])
            for area_id in affected_area_ids
        }
        temp_context = self._context_after_candidate(context, group, candidate)
        group_count_by_area = self._distinct_group_count_by_area(partial)

        competition_penalty = 0.0
        fragment_penalty = 0.0
        headroom_penalty = 0.0
        for area_id in affected_area_ids:
            state_after = temp_context.states[area_id]
            n_groups_after = group_count_by_area.get(area_id, 0) + 1
            n_segments = self._stage2_proxy_segment_count(state_after, group)
            competition_penalty += n_groups_after / max(1, n_segments)
            fragment_penalty += max(
                0.0,
                self._area_fragmentation_score(state_after)
                - before_fragment.get(area_id, 0.0),
            )

        for placement in candidate.placements:
            state_after = temp_context.states[placement.area_id]
            headroom = self._stage2_proxy_headroom(state_after, group)
            demand = max(1, placement.container_count)
            shortage = max(0.0, demand - headroom)
            if placement.container_count >= (
                self.config.guard_large_group_column_threshold
                * MAX_TIERS_PER_COLUMN
            ):
                shortage *= 1.0 + shortage / demand
            headroom_penalty += shortage

        split_penalty = max(0, len(candidate.placements) - 1)
        return (
            candidate.delta_cost
            + self.config.guard_competition_weight * competition_penalty
            + self.config.guard_fragment_weight * fragment_penalty
            + self.config.guard_headroom_weight * headroom_penalty
            + self.config.guard_split_weight * split_penalty
        )

    def _filter_stage2_guarded_candidates(
        self,
        candidates: List[Stage1AssignmentCandidate],
        group: AllocationGroup,
        partial: Stage1Solution,
        context: Stage1CostContext,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        base_cost: float,
    ) -> List[Stage1AssignmentCandidate]:
        del group_by_id, yard_areas, base_cost
        group_count_by_area = self._distinct_group_count_by_area(partial)
        filtered: List[Stage1AssignmentCandidate] = []
        for candidate in candidates:
            temp_context = self._context_after_candidate(context, group, candidate)
            if self._stage2_guarded_candidate_passes(
                candidate,
                group,
                temp_context,
                group_count_by_area,
            ):
                filtered.append(candidate)
        return filtered

    def _stage2_guarded_candidate_passes(
        self,
        candidate: Stage1AssignmentCandidate,
        group: AllocationGroup,
        context_after: Stage1CostContext,
        group_count_by_area: Dict[str, int],
    ) -> bool:
        affected_area_ids = {placement.area_id for placement in candidate.placements}
        for area_id in affected_area_ids:
            state_after = context_after.states[area_id]
            n_groups_after = group_count_by_area.get(area_id, 0) + 1
            n_segments = self._stage2_proxy_segment_count(state_after, group)
            competition = n_groups_after / max(1, n_segments)
            if competition > self.config.guard_max_competition_ratio:
                return False

        for placement in candidate.placements:
            if placement.container_count < (
                self.config.guard_large_group_column_threshold
                * MAX_TIERS_PER_COLUMN
            ):
                continue
            headroom = self._stage2_proxy_headroom(
                context_after.states[placement.area_id],
                group,
            )
            if headroom < self.config.guard_min_headroom_ratio * placement.container_count:
                return False
        return True

    def _context_after_candidate(
        self,
        context: Stage1CostContext,
        group: AllocationGroup,
        candidate: Stage1AssignmentCandidate,
    ) -> Stage1CostContext:
        temp_context = context.clone()
        for placement, preview_entry in zip(
            candidate.placements,
            candidate.previews,
        ):
            _area_id, preview = preview_entry
            self._apply_preview_to_context(
                temp_context,
                group,
                placement,
                preview,
            )
        return temp_context

    @staticmethod
    def _distinct_group_count_by_area(partial: Stage1Solution) -> Dict[str, int]:
        group_ids_by_area: Dict[str, Set[str]] = defaultdict(set)
        for group_id, placements in partial.iter_group_placements():
            for placement in placements:
                group_ids_by_area[placement.area_id].add(group_id)
        return {
            area_id: len(group_ids)
            for area_id, group_ids in group_ids_by_area.items()
        }

    def _stage2_proxy_segment_count(
        self,
        state: AreaResourceState,
        group: AllocationGroup,
    ) -> int:
        if group.size == ContainerSize.SIZE_20:
            return sum(
                1
                for bay_number, free_tiers in state.bay_free_tiers.items()
                if free_tiers > 0 and state._can_use_bay_for_20ft(bay_number)
            )
        return sum(
            1
            for bay_a, bay_b, is_edge in state.pair_bays
            if (group.size != ContainerSize.SIZE_45 or is_edge)
            and state._can_use_pair_for_large(bay_a, bay_b)
            and min(
                state.bay_free_tiers.get(bay_a, 0),
                state.bay_free_tiers.get(bay_b, 0),
            ) > 0
        )

    def _stage2_proxy_headroom(
        self,
        state: AreaResourceState,
        group: AllocationGroup,
    ) -> int:
        if group.size == ContainerSize.SIZE_20:
            return self._contiguous_20ft_headroom(state)
        return self._contiguous_large_headroom(
            state,
            edge_only=group.size == ContainerSize.SIZE_45,
        )

    @staticmethod
    def _contiguous_20ft_headroom(state: AreaResourceState) -> int:
        best = 0
        current = 0
        previous_bay: Optional[int] = None
        for bay_number in sorted(state.bay_free_tiers):
            free_tiers = state.bay_free_tiers.get(bay_number, 0)
            if free_tiers <= 0 or not state._can_use_bay_for_20ft(bay_number):
                current = 0
                previous_bay = None
                continue
            if previous_bay is None or bay_number == previous_bay + 1:
                current += free_tiers
            else:
                current = free_tiers
            best = max(best, current)
            previous_bay = bay_number
        return best

    @staticmethod
    def _contiguous_large_headroom(
        state: AreaResourceState,
        edge_only: bool,
    ) -> int:
        best = 0
        current = 0
        previous_pair: Optional[Tuple[int, int]] = None
        for bay_a, bay_b, is_edge in sorted(state.pair_bays):
            if edge_only and not is_edge:
                current = 0
                previous_pair = None
                continue
            free_tiers = min(
                state.bay_free_tiers.get(bay_a, 0),
                state.bay_free_tiers.get(bay_b, 0),
            )
            if free_tiers <= 0 or not state._can_use_pair_for_large(bay_a, bay_b):
                current = 0
                previous_pair = None
                continue
            if previous_pair is None or bay_a <= previous_pair[1] + 1:
                current += free_tiers
            else:
                current = free_tiers
            best = max(best, current)
            previous_pair = (bay_a, bay_b)
        return best

    @staticmethod
    def _repair_group_difficulty(group: AllocationGroup) -> float:
        if group.size == ContainerSize.SIZE_45:
            size_difficulty = 300.0 if group.is_edge_only else 250.0
        elif group.size == ContainerSize.SIZE_40:
            size_difficulty = 200.0
        else:
            size_difficulty = 100.0
        split_difficulty = 25.0 if group.is_split else 0.0
        return (
            size_difficulty
            + AreaResourceState._group_tier_demand(group)
            + split_difficulty
        )

    def _placement_container_counts(
        self,
        group: AllocationGroup,
        placements: List[Stage1Placement],
    ) -> List[float]:
        total_containers = max(0, group.container_count)
        if not placements:
            return []
        if len(placements) == 1:
            return [float(total_containers)]
        if total_containers <= 0:
            return [0.0 for _ in placements]
        total_tier_demand = self._group_tier_demand(group)
        if total_tier_demand <= 0:
            counts = [0.0 for _ in placements]
            counts[0] = float(total_containers)
            return counts

        remaining = total_containers
        counts: List[float] = []
        for index, placement in enumerate(placements):
            if index == len(placements) - 1:
                count = remaining
            else:
                count = max(
                    0,
                    round(
                        group.container_count
                        * placement.container_count
                        / total_tier_demand
                    ),
                )
                count = min(count, remaining)
            counts.append(float(count))
            remaining = max(0, remaining - count)
        return counts

    def _planning_capacity(self, area_id: str, states: Dict[str, AreaResourceState]) -> int:
        capacity = max(1, states[area_id].initial_total_tiers)
        area = self._area_by_id.get(area_id)
        capacity_factor = self._area_busy_capacity_factor(area)
        return max(
            1,
            int(
                math.floor(
                    capacity
                    * self.config.workload_soft_capacity_ratio
                    * capacity_factor
                )
            ),
        )

    def _planning_remaining_capacity(
        self,
        area_id: str,
        area_load: Dict[str, int],
        states: Dict[str, AreaResourceState],
    ) -> int:
        return max(0, self._planning_capacity(area_id, states) - area_load.get(area_id, 0))

    def _log_unassigned_diagnostics(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
    ) -> None:
        if not solution.unassigned_group_ids:
            return

        context = self._replay_solution(
            solution,
            group_by_id,
            yard_areas,
        )
        for group_id in sorted(
            solution.unassigned_group_ids,
            key=lambda gid: self._difficulty_key(group_by_id[gid], yard_areas)
            if gid in group_by_id
            else (999, 999, 999, 999),
        ):
            group = group_by_id.get(group_id)
            if group is None:
                continue
            logger.info(
                "Stage1 unassigned diagnostic: group=%s size=%s demand=%s line=%s reason=%s",
                group.group_id,
                getattr(group.size, "value", group.size),
                self._group_tier_demand(group),
                group.line_key,
                self._unassigned_failure_reason(
                    solution,
                    group,
                    group_by_id,
                    yard_areas,
                    context,
                ),
            )

    def _unassigned_failure_reason(
        self,
        solution: Stage1Solution,
        group: AllocationGroup,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        context: Stage1CostContext,
    ) -> str:
        compatible_areas = [
            area for area in yard_areas if ConstraintChecker.can_assign_to_area(group, area)
        ]
        if not compatible_areas:
            return "no compatible area"

        capacity_areas = [
            area
            for area in compatible_areas
            if self._group_tier_demand(group)
            <= self._planning_remaining_capacity(area.area_id, context.area_load, context.states)
        ]
        distribution_candidates = self._generate_repair_candidates(
            solution,
            group,
            group_by_id,
            yard_areas,
            context,
            self.evaluate(solution, group_by_id, yard_areas),
        )
        if not capacity_areas:
            if distribution_candidates:
                return "multi-area distribution feasible but was not selected or accepted"
            max_remaining = max(
                self._planning_remaining_capacity(area.area_id, context.area_load, context.states)
                for area in compatible_areas
            )
            return f"insufficient planning capacity max_remaining={max_remaining}"

        preview_failures: Dict[str, int] = defaultdict(int)
        for area in capacity_areas:
            preview = context.states[area.area_id].preview_place(group)
            if preview.feasible:
                return "feasible full-area candidate exists but was not selected or accepted"
            preview_failures[preview.reason or "preview infeasible"] += 1

        if distribution_candidates:
            return "multi-area distribution feasible but was not selected or accepted"

        if preview_failures:
            reason, count = max(preview_failures.items(), key=lambda item: item[1])
            return f"preview infeasible: {reason} ({count} areas)"
        return "distribution infeasible"

    def _line_concentration_cost(
        self,
        line_area_load: Dict[Tuple[Optional[int], str], int],
    ) -> float:
        total_by_line: Dict[Optional[int], int] = defaultdict(int)
        areas_by_line: Dict[Optional[int], Dict[str, int]] = defaultdict(dict)

        for (line_key, area_id), load in line_area_load.items():
            if load <= 0:
                continue
            total_by_line[line_key] += load
            areas_by_line[line_key][area_id] = areas_by_line[line_key].get(area_id, 0) + load

        cost = 0.0
        for line_key, total in total_by_line.items():
            area_loads = areas_by_line[line_key]
            if total >= self.config.line_min_total_for_share_penalty:
                threshold = self.config.line_max_area_share_threshold
                share_denominator = max(1e-6, 1.0 - threshold)
                for load in area_loads.values():
                    share = load / total
                    excess = max(0.0, share - threshold)
                    if excess > 0.0:
                        cost += self.config.line_share_penalty_weight * (
                            excess / share_denominator
                        ) ** self.config.line_share_penalty_power
            small_threshold = max(2, math.ceil(total * 0.12))
            for load in area_loads.values():
                if load < small_threshold:
                    cost += self.config.line_small_fragment_weight * (
                        (small_threshold - load) / small_threshold
                    )
        return cost

    def _placement_berth_distance_cost(
        self,
        group: AllocationGroup,
        area: Optional[YardArea],
        column_demand: int,
    ) -> float:
        if area is None or column_demand <= 0:
            return 0.0
        area_coord = area.center_coordinate
        vessel = self._vessels.get(group.voyage_id)
        berth_coord = vessel.berth_coordinate if vessel is not None else None
        if area_coord is None or berth_coord is None:
            return 0.0

        distance = math.hypot(
            float(area_coord[0]) - float(berth_coord[0]),
            float(area_coord[1]) - float(berth_coord[1]),
        )
        if vessel is not None:
            area.distance_to_berth[vessel.voyage_id] = distance
        return (distance / 1000.0) * float(column_demand)

    @staticmethod
    def _area_busy_capacity_factor(area: Optional[YardArea]) -> float:
        if area is None or area.business_type != BusinessType.EXPORT:
            return 1.0
        factor = getattr(area, "_stage1_capacity_factor", 1.0)
        try:
            factor_value = float(factor)
        except (TypeError, ValueError):
            return 1.0
        return min(1.0, max(0.2, factor_value))

    @staticmethod
    def _placement_busy_profile_cost(
        group: AllocationGroup,
        area: Optional[YardArea],
        column_demand: int,
    ) -> float:
        if (
            area is None
            or column_demand <= 0
            or group.business_type != BusinessType.EXPORT
        ):
            return 0.0
        peak_busy_ratio = getattr(area, "_stage1_peak_busy_ratio", 0.0)
        try:
            peak = float(peak_busy_ratio)
        except (TypeError, ValueError):
            return 0.0
        if peak <= 0.0:
            return 0.0
        return float(column_demand) * peak * peak

    def _difficulty_key(
        self,
        group: AllocationGroup,
        yard_areas: List[YardArea],
    ) -> Tuple[int, int, int, int]:
        candidate_count = sum(
            1 for area in yard_areas if ConstraintChecker.can_assign_to_area(group, area)
        )
        size_rank = {
            ContainerSize.SIZE_45: 0,
            ContainerSize.SIZE_40: 1,
            ContainerSize.SIZE_20: 2,
        }.get(group.size, 3)
        return (
            candidate_count,
            size_rank,
            -self._group_tier_demand(group),
            0 if group.is_edge_only else 1,
        )

    def _materialize_solution(
        self,
        solution: Stage1Solution,
        groups: List[AllocationGroup],
        area_by_id: Dict[str, YardArea],
    ) -> Tuple[List[AreaAssignment], List[AllocationGroup], List[AllocationGroup]]:
        group_by_id = {group.group_id: group for group in groups}
        solution.validate_conservation(
            {
                group_id: self._group_tier_demand(group)
                for group_id, group in group_by_id.items()
            }
        )
        assignments: List[AreaAssignment] = []
        unassigned: List[AllocationGroup] = []
        generated_groups: List[AllocationGroup] = []

        for group in groups:
            placements = solution.group_placements(group.group_id)
            if not placements:
                unassigned.append(group)
                continue
            if len(placements) == 1:
                placement = placements[0]
                assignments.append(
                    AreaAssignment(
                        assignment_id=f"ASN-{uuid.uuid4().hex[:8].upper()}",
                        group_id=group.group_id,
                        yard_area_id=placement.area_id,
                        column_demand=self._tier_demand_to_columns(
                            placement.container_count
                        ),
                        split_index=0,
                        is_partial=False,
                    )
                )
                continue

            split_groups = self._split_group_for_placements(group, placements)
            generated_groups.extend(split_groups)
            for split_group, placement in zip(split_groups, placements):
                assignments.append(
                    AreaAssignment(
                        assignment_id=f"ASN-{uuid.uuid4().hex[:8].upper()}",
                        group_id=split_group.group_id,
                        yard_area_id=placement.area_id,
                        column_demand=self._tier_demand_to_columns(
                            placement.container_count
                        ),
                        split_index=split_group.split_index,
                        is_partial=True,
                    )
                )

        return assignments, unassigned, generated_groups

    def _split_group_for_placements(
        self,
        group: AllocationGroup,
        placements: List[Stage1Placement],
    ) -> List[AllocationGroup]:
        containers = list(group.containers)
        total_demand = max(1, self._group_tier_demand(group))
        total_containers = max(group.container_count, len(containers))
        remaining_container_count = total_containers
        split_groups: List[AllocationGroup] = []

        for idx, placement in enumerate(placements, start=1):
            if idx < len(placements):
                count = max(
                    0,
                    round(total_containers * placement.container_count / total_demand),
                )
                count = min(count, remaining_container_count)
            else:
                count = remaining_container_count

            assigned_container_count = min(count, len(containers))
            sub_containers = containers[:assigned_container_count]
            containers = containers[assigned_container_count:]
            remaining_container_count = max(0, remaining_container_count - count)
            sub_group = AllocationGroup(
                group_id=f"{group.group_id}-S{idx}",
                business_type=group.business_type,
                size=group.size,
                container_type=group.container_type,
                weight_class=group.weight_class,
                voyage_id=group.voyage_id,
                line_key=group.line_key,
                group_attributes=group.group_attributes.copy(),
                containers=sub_containers,
                container_count=count,
                column_demand=self._tier_demand_to_columns(
                    placement.container_count
                ),
                earliest_arrival=group.earliest_arrival,
                latest_departure=group.latest_departure,
                is_split=True,
                parent_group_id=group.group_id,
                split_index=idx,
            )
            sub_group.group_attributes.update(group.group_attributes)
            split_groups.append(sub_group)

        return split_groups


class Stage2BayAllocator:
    """
    Stage 2: within each assigned yard area, decide concrete bay/column usage.

    The implementation delegates to the greedy allocator in `stage2_scip` while
    keeping the historical public interface unchanged.

    Sibling split groups that share the same parent and yard area are merged into
    one Stage2 item before placement so bay-axis fragmentation penalties apply to a
    single demand block (see `merge_same_parent_assignments_for_stage2`).

    Returns ``(bay_allocations, merged_area_assignments)`` — the latter replaces
    sibling rows with one combined assignment per parent so it stays consistent
    with Stage2 output.
    """

    def __init__(self, scip_config=None):
        self.scip_config = scip_config
        self.last_placement_progress: List[Dict[str, Any]] = []

    def allocate(
        self,
        area_assignments: List[AreaAssignment],
        groups: Dict[str, AllocationGroup],
        yard_areas: Dict[str, YardArea],
    ) -> Tuple[List[BayColumnAllocation], List[AreaAssignment]]:
        from yardplan_core.stage2 import (
            ScipStage2BayAllocator,
            log_stage2_merged_parent_contiguity,
            merge_same_parent_assignments_for_stage2,
        )

        merged_assignments, merged_keys = merge_same_parent_assignments_for_stage2(
            area_assignments,
            groups,
        )
        allocator = ScipStage2BayAllocator(self.scip_config)
        allocations = allocator.allocate(
            merged_assignments,
            groups,
            yard_areas,
        )
        self.last_placement_progress = list(getattr(allocator, "placement_progress", []))
        log_stage2_merged_parent_contiguity(allocations, merged_keys)
        return allocations, merged_assignments


class AllocationEngine:
    """Orchestrates stage 1 + existing stage 2 for Function 2."""

    def __init__(
        self,
        stage1: Optional[Stage1YardAreaAssigner] = None,
        stage2: Optional[Stage2BayAllocator] = None,
    ):
        self.stage1 = stage1 or Stage1YardAreaAssigner()
        self.stage2 = stage2 or Stage2BayAllocator()
        self.last_stage1_convergence: List[Dict[str, Any]] = []
        self.last_stage2_progress: List[Dict[str, Any]] = []

    def allocate(
        self,
        groups: List[AllocationGroup],
        yard_areas: List[YardArea],
        workload_snapshot: Optional[AreaWorkloadSnapshot] = None,
    ) -> Tuple[List[AreaAssignment], List[BayColumnAllocation], List[AllocationGroup]]:
        self.stage1.set_workload_snapshot(workload_snapshot)
        area_assignments, unassigned = self.stage1.assign(groups, yard_areas)
        group_dict = {group.group_id: group for group in groups}
        bay_allocations, area_assignments = self.stage2.allocate(
            area_assignments,
            group_dict,
            {area.area_id: area for area in yard_areas},
        )
        self.last_stage1_convergence = list(
            getattr(self.stage1, "_last_convergence_history", [])
        )
        self.last_stage2_progress = list(
            getattr(self.stage2, "last_placement_progress", [])
        )
        return area_assignments, bay_allocations, unassigned


__all__ = [
    "TimeStep",
    "RollingWindow",
    "RollingWindowPlanner",
    "ConstraintChecker",
    "PlacementPreview",
    "AreaResourceState",
    "Stage1LNSConfig",
    "Stage1CostBreakdown",
    "Stage1DerivedStats",
    "AreaScoringStrategy",
    "DefaultAreaScoringStrategy",
    "Stage1YardAreaAssigner",
    "Stage2BayAllocator",
    "AllocationEngine",
]
