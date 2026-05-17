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
    Vessel,
    YardArea,
    logger,
)
from yardplan_core.workload import (
    AreaWorkloadProvider,
    AreaWorkloadSnapshot,
    SimulatedAreaWorkloadProvider,
    WorkloadEstimationConfig,
    print_eta_step_workloads,
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

    It tracks free columns by physical bay number and simulates the shared
    resource conflict between 20ft bay usage and 40/45ft pair usage.
    """

    area_id: str
    bay_free: Dict[int, int]
    pair_bays: List[Tuple[int, int, bool]]
    locked_20_bays: Set[int]
    locked_large_bays: Set[int]
    initial_total_columns: int
    initial_large_columns: int
    initial_edge_columns: int

    @classmethod
    def from_area(cls, area: YardArea) -> "AreaResourceState":
        bay_free: Dict[int, int] = {}
        locked_20_bays = set(getattr(area, "_stage2_existing_20ft_bays", set()))
        locked_large_bays = set(getattr(area, "_stage2_existing_large_bays", set()))
        for bay in area.bays:
            if bay.can_accept_20ft():
                bay_free[bay.bay_number] = max(0, bay.free_columns)

        pair_bays: List[Tuple[int, int, bool]] = []
        for pair in area.large_bay_pairs:
            a = pair.bay_a.bay_number
            b = pair.bay_b.bay_number
            pair_free = max(0, pair.free_columns)
            bay_free[a] = min(bay_free.get(a, pair_free), pair_free)
            bay_free[b] = min(bay_free.get(b, pair_free), pair_free)
            pair_bays.append((a, b, pair.is_edge_pair))

        initial_total = sum(
            free_columns
            for bay_number, free_columns in bay_free.items()
            if bay_number not in locked_large_bays
        )
        initial_large = sum(
            min(bay_free.get(a, 0), bay_free.get(b, 0))
            for a, b, _ in pair_bays
            if cls._large_pair_available(a, b, locked_20_bays, locked_large_bays)
        )
        initial_edge = sum(
            min(bay_free.get(a, 0), bay_free.get(b, 0))
            for a, b, is_edge in pair_bays
            if is_edge and cls._large_pair_available(a, b, locked_20_bays, locked_large_bays)
        )
        return cls(
            area_id=area.area_id,
            bay_free=bay_free,
            pair_bays=pair_bays,
            locked_20_bays=locked_20_bays,
            locked_large_bays=locked_large_bays,
            initial_total_columns=max(1, initial_total),
            initial_large_columns=max(1, initial_large),
            initial_edge_columns=max(1, initial_edge),
        )

    def clone(self) -> "AreaResourceState":
        return AreaResourceState(
            area_id=self.area_id,
            bay_free=dict(self.bay_free),
            pair_bays=list(self.pair_bays),
            locked_20_bays=set(self.locked_20_bays),
            locked_large_bays=set(self.locked_large_bays),
            initial_total_columns=self.initial_total_columns,
            initial_large_columns=self.initial_large_columns,
            initial_edge_columns=self.initial_edge_columns,
        )

    def preview_place(
        self,
        group: AllocationGroup,
        demand: Optional[int] = None,
    ) -> PlacementPreview:
        required = demand if demand is not None else group.column_demand
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
            self.bay_free[bay_number] = max(0, self.bay_free.get(bay_number, 0) - used)
        self.locked_20_bays.update(preview.newly_locked_20_bays)

        for (bay_a, bay_b), used, _is_edge in preview.used_large_pairs:
            self.bay_free[bay_a] = max(0, self.bay_free.get(bay_a, 0) - used)
            self.bay_free[bay_b] = max(0, self.bay_free.get(bay_b, 0) - used)
        self.locked_large_bays.update(preview.newly_locked_large_bays)

    def max_placeable_columns(self, group: AllocationGroup) -> int:
        if group.size == ContainerSize.SIZE_20:
            return sum(
                free_columns
                for bay_number, free_columns in self.bay_free.items()
                if self._can_use_bay_for_20ft(bay_number)
            )
        if group.size == ContainerSize.SIZE_45:
            return self._max_large_placeable_columns(edge_only=True)
        return self._max_large_placeable_columns(edge_only=False)

    def _preview_20ft(self, demand: int) -> PlacementPreview:
        remaining = demand
        used: List[Tuple[int, int]] = []
        candidates = sorted(
            (
                (bay_number, free_columns)
                for bay_number, free_columns in self.bay_free.items()
                if self._can_use_bay_for_20ft(bay_number)
            ),
            key=lambda item: (-item[1], item[0]),
        )

        for bay_number, free_columns in candidates:
            if remaining <= 0:
                break
            if free_columns <= 0:
                continue
            take = min(free_columns, remaining)
            used.append((bay_number, take))
            remaining -= take

        if remaining > 0:
            return PlacementPreview(False, demand, reason="insufficient 20ft bay columns")

        leftover_small = sum(
            1
            for bay_number, used_columns in used
            if 0 < self.bay_free.get(bay_number, 0) - used_columns <= 1
        )
        fragmentation = (len(used) - 1) * 1.5 + leftover_small * 0.8
        large_after = self._remaining_large_columns_after_20ft(used)
        large_loss_ratio = max(0.0, 1.0 - large_after / self.initial_large_columns)
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
            free_columns = min(self.bay_free.get(bay_a, 0), self.bay_free.get(bay_b, 0))
            if free_columns > 0:
                candidates.append((bay_a, bay_b, is_edge, free_columns))

        candidates.sort(key=lambda item: (item[2], -item[3], item[0], item[1]))

        for bay_a, bay_b, is_edge, free_columns in candidates:
            if remaining <= 0:
                break
            if not self._large_pair_available(
                bay_a,
                bay_b,
                self.locked_20_bays,
                simulated_locked_large_bays,
            ):
                continue
            take = min(free_columns, remaining)
            used.append(((bay_a, bay_b), take, is_edge))
            remaining -= take
            simulated_locked_large_bays.update((bay_a, bay_b))

        if remaining > 0:
            resource_name = "edge large-bay columns" if edge_only else "large-bay columns"
            return PlacementPreview(False, demand, reason=f"insufficient {resource_name}")

        leftover_small = 0
        for (bay_a, bay_b), used_columns, _is_edge in used:
            left = min(self.bay_free.get(bay_a, 0), self.bay_free.get(bay_b, 0)) - used_columns
            if 0 < left <= 1:
                leftover_small += 1

        edge_used = sum(columns for _pair, columns, is_edge in used if is_edge)
        total_used = sum(columns for _pair, columns, _is_edge in used)
        fragmentation = (len(used) - 1) * 2.0 + leftover_small
        scarcity = total_used / self.initial_large_columns
        if edge_only:
            scarcity += total_used / self.initial_edge_columns * 2.0
        elif edge_used:
            scarcity += edge_used / self.initial_edge_columns * 1.5

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

    def _remaining_large_columns_after_20ft(self, used_20ft: List[Tuple[int, int]]) -> int:
        simulated = dict(self.bay_free)
        for bay_number, used_columns in used_20ft:
            simulated[bay_number] = max(0, simulated.get(bay_number, 0) - used_columns)
        simulated_locked_20_bays = set(self.locked_20_bays)
        simulated_locked_20_bays.update(bay_number for bay_number, _used in used_20ft)
        return self._max_large_columns_for_locks(
            edge_only=False,
            bay_free=simulated,
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

    def _max_large_placeable_columns(self, edge_only: bool) -> int:
        return self._max_large_columns_for_locks(
            edge_only=edge_only,
            bay_free=self.bay_free,
            locked_20_bays=self.locked_20_bays,
            locked_large_bays=self.locked_large_bays,
        )

    def _max_large_columns_for_locks(
        self,
        edge_only: bool,
        bay_free: Dict[int, int],
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
            free_columns = min(bay_free.get(bay_a, 0), bay_free.get(bay_b, 0))
            if free_columns > 0:
                candidates.append((bay_a, bay_b, is_edge, free_columns))

        total = 0
        candidates.sort(key=lambda item: (item[2], -item[3], item[0], item[1]))
        for bay_a, bay_b, _is_edge, free_columns in candidates:
            if not self._large_pair_available(
                bay_a,
                bay_b,
                locked_20_bays,
                simulated_locked_large_bays,
            ):
                continue
            total += free_columns
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

    max_split_parts: int = 4
    large_group_column_threshold: int = 4
    max_iterations: int = 80
    no_improve_limit: int = 20
    destroy_fraction: float = 0.25
    random_seed: int = 17
    time_limit_seconds: Optional[float] = None
    workload_provider: Optional[str] = "simulated"
    workload_soft_capacity_ratio: float = 0.85
    workload_balance_weight: float = 20.0
    workload_use_target_eta_step_only: bool = True
    workload_non_target_step_weight: float = 0.0
    line_small_fragment_weight: float = 5.0
    split_weight: float = 12.0
    large_group_split_discount: float = 0.45
    small_group_split_premium: float = 1.2
    export_target_containers_per_area: int = 80
    import_target_containers_per_area: int = 120
    min_split_part_containers: int = 25
    group_peak_weight: float = 100.0
    fragment_weight: float = 12.0
    base_split_weight: float = 6.0
    workload_area_soft_target_export: float = 80.0
    workload_area_soft_target_import: float = 120.0
    workload_overload_weight: float = 18.0
    workload_spread_weight_scale: float = 0.15
    physical_weight: float = 7.0
    unassigned_weight: float = 100000.0
    line_max_area_share_threshold: float = 0.40
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


@dataclass
class Stage1Placement:
    group_id: str
    area_id: str
    column_demand: int


@dataclass
class Stage1Solution:
    placements_by_group: Dict[str, List[Stage1Placement]] = field(default_factory=dict)
    unassigned_group_ids: set = field(default_factory=set)

    def clone(self) -> "Stage1Solution":
        return Stage1Solution(
            placements_by_group={
                group_id: [
                    Stage1Placement(p.group_id, p.area_id, p.column_demand)
                    for p in placements
                ]
                for group_id, placements in self.placements_by_group.items()
            },
            unassigned_group_ids=set(self.unassigned_group_ids),
        )


@dataclass
class Stage1CostContext:
    states: Dict[str, AreaResourceState]
    area_load: Dict[str, int]
    line_area_load: Dict[Tuple[Optional[int], str], int]
    workload_overlay: Dict[Tuple[str, int], Tuple[float, float]]
    physical_cost: float = 0.0
    infeasible_count: int = 0

    def clone(self) -> "Stage1CostContext":
        return Stage1CostContext(
            states={area_id: state.clone() for area_id, state in self.states.items()},
            area_load=defaultdict(int, self.area_load),
            line_area_load=defaultdict(int, self.line_area_load),
            workload_overlay=dict(self.workload_overlay),
            physical_cost=self.physical_cost,
            infeasible_count=self.infeasible_count,
        )


@dataclass
class Stage1AssignmentCandidate:
    delta_cost: float
    placements: List[Stage1Placement]
    previews: List[Tuple[str, PlacementPreview]]


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
        projected = current_assignments.get(area.area_id, 0) + group.column_demand
        return -projected


class Stage1YardAreaAssigner:
    """
    Stage 1: assign allocation groups to one or more yard areas.

    The implementation uses a constraint-aware initial solution followed by a
    small LNS loop with optional simulated annealing. It only produces
    AreaAssignment and unassigned groups; exact
    bay/range allocation remains a second-stage concern.

    Workload balance is measured in RTG moves (箱次), not columns. Baseline
    moves come from `AreaWorkloadSnapshot`; planned overlay comes from
    `AllocationGroup.container_count`, evenly spread across the voyage's
    ETA~ETD step span. Capacity and workload stay decoupled: `column_demand`
    is still used for yard capacity, but never as the workload metric.
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
        self.workload_provider = workload_provider or self._resolve_workload_provider()
        self.workload_estimation_config = (
            workload_estimation_config or WorkloadEstimationConfig()
        )
        self.workload_snapshot: Optional[AreaWorkloadSnapshot] = None
        self._vessels: Dict[str, Vessel] = {}
        self._time_steps: List[TimeStep] = []
        self._voyage_step_span: Dict[str, List[int]] = {}

    def _resolve_workload_provider(self) -> AreaWorkloadProvider:
        provider_name = (self.config.workload_provider or "simulated").lower()
        if provider_name == "simulated":
            return SimulatedAreaWorkloadProvider()
        raise ValueError(f"未知 workload_provider: {self.config.workload_provider!r}")

    def set_workload_snapshot(
        self,
        snapshot: Optional[AreaWorkloadSnapshot],
    ) -> None:
        self.workload_snapshot = snapshot
        if snapshot is None:
            self._reset_workload_context()
            return
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
        if not time_steps:
            self.workload_snapshot = None
            self._reset_workload_context()
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
        self._cache_workload_context(vessels=vessels, time_steps=time_steps)
        return snapshot

    def _reset_workload_context(self) -> None:
        self._vessels = {}
        self._time_steps = []
        self._voyage_step_span = {}

    def _cache_workload_context(
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

        logger.info(
            "Stage1 LNS complete: %s assignments, %s split groups, %s unassigned",
            len(assignments),
            len(generated_groups),
            len(unassigned),
        )
        return assignments, unassigned

    def _build_states(self, yard_areas: List[YardArea]) -> Dict[str, AreaResourceState]:
        return {area.area_id: AreaResourceState.from_area(area) for area in yard_areas}

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
                solution.placements_by_group.pop(group.group_id, None)
                solution.unassigned_group_ids.add(group.group_id)
                continue
            solution.placements_by_group[group.group_id] = candidate.placements
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

        for _iteration in range(self.config.max_iterations):
            if self.config.time_limit_seconds is not None:
                if time.monotonic() - started_at >= self.config.time_limit_seconds:
                    break
            if no_improve >= self.config.no_improve_limit:
                break

            removed_ids = self._random_destroy(current)
            if not removed_ids:
                break

            candidate = current.clone()
            for group_id in removed_ids:
                candidate.placements_by_group.pop(group_id, None)
                candidate.unassigned_group_ids.discard(group_id)

            repaired = self._random_repair(
                candidate,
                removed_ids,
                group_by_id,
                yard_areas,
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

        self._log_workload_balance(best, group_by_id, yard_areas)
        self._log_unassigned_diagnostics(best, group_by_id, yard_areas)
        if self.config.use_simulated_annealing:
            logger.info(
                "Stage1 LNS best cost: %.3f, unassigned=%s (SA final T=%.4f)",
                best_cost,
                len(best.unassigned_group_ids),
                temperature,
            )
        else:
            logger.info(
                "Stage1 LNS best cost: %.3f, unassigned=%s",
                best_cost,
                len(best.unassigned_group_ids),
            )
        return best

    def evaluate(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
    ) -> float:
        context = self._replay_solution(solution, group_by_id, yard_areas)
        cost = (context.infeasible_count + len(solution.unassigned_group_ids)) * self.config.unassigned_weight
        cost += self._time_step_workload_cost(context, yard_areas)
        cost += self._line_concentration_cost(context.line_area_load)
        cost += self._group_dispersion_cost(solution, group_by_id)
        cost += self._split_operation_cost(solution, group_by_id)
        cost += context.physical_cost * self.config.physical_weight
        return cost

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
        before_cost = base_cost if base_cost is not None else self.evaluate(partial, group_by_id, yard_areas)
        candidate = partial.clone()
        candidate.placements_by_group[group.group_id] = [
            Stage1Placement(p.group_id, p.area_id, p.column_demand) for p in placements
        ]
        candidate.unassigned_group_ids.discard(group.group_id)
        after_cost = self.evaluate(candidate, group_by_id, yard_areas)
        return after_cost - before_cost

    def _random_destroy(self, solution: Stage1Solution) -> List[str]:
        assigned_ids = list(solution.placements_by_group.keys())
        if not assigned_ids:
            return []
        remove_count = max(1, int(math.ceil(len(assigned_ids) * self.config.destroy_fraction)))
        return self._rng.sample(assigned_ids, min(remove_count, len(assigned_ids)))

    def _random_repair(
        self,
        partial: Stage1Solution,
        group_ids: List[str],
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
    ) -> Stage1Solution:
        ordered = [group_by_id[group_id] for group_id in group_ids if group_id in group_by_id]
        self._rng.shuffle(ordered)
        for group in ordered:
            candidate = self._try_assign_group(partial, group, group_by_id, yard_areas)
            if candidate is None:
                partial.placements_by_group.pop(group.group_id, None)
                partial.unassigned_group_ids.add(group.group_id)
                continue
            partial.placements_by_group[group.group_id] = candidate.placements
            partial.unassigned_group_ids.discard(group.group_id)
        return partial

    def _effective_split_weight(self, group: AllocationGroup) -> float:
        total_containers = max(1, group.container_count)
        target = self._target_containers_per_area(group)
        size_factor = min(1.0, target / total_containers)
        if total_containers <= 100:
            group_class_factor = 1.4
        elif total_containers <= 300:
            group_class_factor = 1.0
        else:
            group_class_factor = 0.7
        return self.config.base_split_weight * size_factor * group_class_factor

    def _try_assign_group(
        self,
        partial: Stage1Solution,
        group: AllocationGroup,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
    ) -> Optional[Stage1AssignmentCandidate]:
        context = self._replay_solution(partial, group_by_id, yard_areas)
        base_cost = self.evaluate(partial, group_by_id, yard_areas)
        candidates = self._full_assignment_candidates(
            partial,
            group,
            group_by_id,
            yard_areas,
            context,
            base_cost,
        )
        candidates.extend(
            self._proactive_split_candidates(
                partial,
                group,
                group_by_id,
                yard_areas,
                context,
                base_cost,
            )
        )
        if not candidates:
            split_candidate = self._repair_group_with_split(
                partial,
                group,
                group_by_id,
                yard_areas,
                context,
                base_cost,
            )
            if split_candidate is not None:
                candidates.append(split_candidate)
        if not candidates:
            return None
        required_areas = self._min_required_area_count(group)
        if required_areas > 1:
            multi_area = [
                candidate
                for candidate in candidates
                if len({placement.area_id for placement in candidate.placements})
                >= required_areas
            ]
            if multi_area:
                candidates = multi_area
        return self._select_repair_candidate(candidates, group)

    def _full_assignment_candidates(
        self,
        partial: Stage1Solution,
        group: AllocationGroup,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        context: Stage1CostContext,
        base_cost: float,
    ) -> List[Stage1AssignmentCandidate]:
        full_candidates: List[Stage1AssignmentCandidate] = []
        for area in yard_areas:
            if not ConstraintChecker.can_assign_to_area(group, area):
                continue
            if (
                group.column_demand
                > self._planning_remaining_capacity(area.area_id, context.area_load, context.states)
            ):
                continue
            preview = context.states[area.area_id].preview_place(group)
            if not preview.feasible:
                continue
            placements = [Stage1Placement(group.group_id, area.area_id, group.column_demand)]
            delta_cost = self.evaluate_placement_delta(
                partial,
                group,
                placements,
                [(area.area_id, preview)],
                group_by_id,
                yard_areas,
                base_cost=base_cost,
            )
            if math.isinf(delta_cost):
                continue
            full_candidates.append(
                Stage1AssignmentCandidate(
                    delta_cost=delta_cost,
                    placements=placements,
                    previews=[(area.area_id, preview)],
                )
            )
        full_candidates.sort(key=lambda candidate: self._candidate_sort_key(candidate, group))
        return full_candidates

    def _proactive_split_candidates(
        self,
        partial: Stage1Solution,
        group: AllocationGroup,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        context: Stage1CostContext,
        base_cost: float,
    ) -> List[Stage1AssignmentCandidate]:
        max_parts = min(
            self.config.max_split_parts,
            group.column_demand,
            sum(
                1
                for area in yard_areas
                if ConstraintChecker.can_assign_to_area(group, area)
            ),
        )
        if max_parts < 2:
            return []

        target = self._target_containers_per_area(group)
        desired_parts = max(2, math.ceil(max(1, group.container_count) / target))
        candidate_part_counts = sorted(set(range(2, min(max_parts, desired_parts + 1) + 1)))
        candidates: List[Stage1AssignmentCandidate] = []
        seen_shapes: Set[Tuple[Tuple[str, int], ...]] = set()

        for part_count in candidate_part_counts:
            column_demands = self._balanced_column_demands(
                group.column_demand,
                part_count,
            )
            candidate = self._build_balanced_split_candidate(
                partial,
                group,
                group_by_id,
                yard_areas,
                context,
                base_cost,
                column_demands,
            )
            if candidate is None or len(candidate.placements) <= 1:
                continue
            shape = tuple(
                sorted(
                    (placement.area_id, placement.column_demand)
                    for placement in candidate.placements
                )
            )
            if shape in seen_shapes:
                continue
            seen_shapes.add(shape)
            candidates.append(candidate)

        candidates.sort(key=lambda candidate: self._candidate_sort_key(candidate, group))
        return candidates

    def _build_balanced_split_candidate(
        self,
        partial: Stage1Solution,
        group: AllocationGroup,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        context: Stage1CostContext,
        base_cost: float,
        column_demands: List[int],
    ) -> Optional[Stage1AssignmentCandidate]:
        placements: List[Stage1Placement] = []
        local_previews: List[Tuple[str, PlacementPreview]] = []
        used_area_ids = set()
        temp_context = context.clone()
        assigned = 0

        for part_columns in column_demands:
            area_candidates: List[Tuple[float, int, str, PlacementPreview]] = []
            for area in yard_areas:
                if area.area_id in used_area_ids:
                    continue
                if not ConstraintChecker.can_assign_to_area(group, area):
                    continue
                if part_columns > self._planning_remaining_capacity(
                    area.area_id,
                    temp_context.area_load,
                    temp_context.states,
                ):
                    continue
                preview = temp_context.states[area.area_id].preview_place(
                    group,
                    demand=part_columns,
                )
                if not preview.feasible:
                    continue

                trial_placements = placements + [
                    Stage1Placement(group.group_id, area.area_id, part_columns)
                ]
                provisional_group_by_id = dict(group_by_id)
                provisional_column_demand = assigned + part_columns
                if group.column_demand > 0:
                    provisional_container_count = max(
                        0,
                        round(
                            group.container_count
                            * provisional_column_demand
                            / group.column_demand
                        ),
                    )
                else:
                    provisional_container_count = max(0, group.container_count)
                provisional_group_by_id[group.group_id] = replace(
                    group,
                    containers=[],
                    column_demand=provisional_column_demand,
                    container_count=provisional_container_count,
                )
                delta_cost = self.evaluate_placement_delta(
                    partial,
                    provisional_group_by_id[group.group_id],
                    trial_placements,
                    local_previews + [(area.area_id, preview)],
                    provisional_group_by_id,
                    yard_areas,
                    base_cost=base_cost,
                )
                if math.isinf(delta_cost):
                    continue
                step_workload = self._area_step_workload(
                    temp_context,
                    area.area_id,
                    self._primary_target_step_id(group),
                )[2]
                area_candidates.append(
                    (
                        delta_cost,
                        step_workload,
                        temp_context.area_load.get(area.area_id, 0),
                        area.area_id,
                        preview,
                    )
                )

            if not area_candidates:
                return None

            area_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            _delta_cost, _step_workload, _area_load, area_id, preview = area_candidates[0]
            temp_context.states[area_id].apply_preview(preview)
            temp_context.area_load[area_id] += part_columns
            temp_context.line_area_load[(group.line_key, area_id)] += part_columns
            temp_context.physical_cost += preview.physical_cost
            assigned += part_columns
            local_previews.append((area_id, preview))
            placements.append(Stage1Placement(group.group_id, area_id, part_columns))
            used_area_ids.add(area_id)

        final_delta = self.evaluate_placement_delta(
            partial,
            group,
            placements,
            local_previews,
            group_by_id,
            yard_areas,
            base_cost=base_cost,
        )
        if math.isinf(final_delta):
            return None
        return Stage1AssignmentCandidate(
            delta_cost=final_delta,
            placements=placements,
            previews=local_previews,
        )

    def _primary_target_step_id(self, group: AllocationGroup) -> int:
        if self.workload_snapshot is None:
            return 0
        target_step_ids = self.workload_snapshot.target_step_ids
        if group.voyage_id in target_step_ids:
            return target_step_ids[group.voyage_id]
        if target_step_ids:
            return next(iter(target_step_ids.values()))
        if self._time_steps:
            return self._time_steps[0].step_id
        return 0

    @staticmethod
    def _balanced_column_demands(total_columns: int, part_count: int) -> List[int]:
        base = total_columns // part_count
        remainder = total_columns % part_count
        return [
            base + (1 if index < remainder else 0)
            for index in range(part_count)
        ]

    def _repair_group_with_split(
        self,
        partial: Stage1Solution,
        group: AllocationGroup,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
        context: Stage1CostContext,
        base_cost: float,
    ) -> Optional[Stage1AssignmentCandidate]:
        remaining = group.column_demand
        assigned = 0
        placements: List[Stage1Placement] = []
        local_previews: List[Tuple[str, PlacementPreview]] = []
        used_area_ids = set()
        temp_context = context.clone()

        while remaining > 0 and len(placements) < self.config.max_split_parts:
            candidates: List[Tuple[float, str, int, PlacementPreview]] = []
            for area in yard_areas:
                if area.area_id in used_area_ids:
                    continue
                if not ConstraintChecker.can_assign_to_area(group, area):
                    continue
                max_columns = min(
                    remaining,
                    temp_context.states[area.area_id].max_placeable_columns(group),
                    self._planning_remaining_capacity(
                        area.area_id,
                        temp_context.area_load,
                        temp_context.states,
                    ),
                )
                if max_columns <= 0:
                    continue
                preview = temp_context.states[area.area_id].preview_place(group, demand=max_columns)
                if not preview.feasible:
                    continue
                trial_placements = placements + [
                    Stage1Placement(group.group_id, area.area_id, max_columns)
                ]
                provisional_group_by_id = dict(group_by_id)
                provisional_column_demand = assigned + max_columns
                if group.column_demand > 0:
                    provisional_container_count = max(
                        0,
                        round(
                            group.container_count
                            * provisional_column_demand
                            / group.column_demand
                        ),
                    )
                else:
                    provisional_container_count = max(0, group.container_count)
                provisional_group_by_id[group.group_id] = replace(
                    group,
                    containers=[],
                    column_demand=provisional_column_demand,
                    container_count=provisional_container_count,
                )
                delta_cost = self.evaluate_placement_delta(
                    partial,
                    provisional_group_by_id[group.group_id],
                    trial_placements,
                    local_previews + [(area.area_id, preview)],
                    provisional_group_by_id,
                    yard_areas,
                    base_cost=base_cost,
                )
                if math.isinf(delta_cost):
                    continue
                candidates.append((delta_cost, area.area_id, max_columns, preview))

            if not candidates:
                return None

            candidates.sort(key=lambda item: (item[0], -item[2], item[1]))
            _delta_cost, area_id, assigned_columns, preview = candidates[0]
            temp_context.states[area_id].apply_preview(preview)
            temp_context.area_load[area_id] += assigned_columns
            temp_context.line_area_load[(group.line_key, area_id)] += assigned_columns
            temp_context.physical_cost += preview.physical_cost
            assigned += assigned_columns
            remaining -= assigned_columns
            local_previews.append((area_id, preview))
            placements.append(Stage1Placement(group.group_id, area_id, assigned_columns))
            used_area_ids.add(area_id)

        if remaining > 0:
            return None
        if len(placements) > self.config.max_split_parts:
            return None

        final_delta = self.evaluate_placement_delta(
            partial,
            group,
            placements,
            local_previews,
            group_by_id,
            yard_areas,
            base_cost=base_cost,
        )
        if math.isinf(final_delta):
            return None
        return Stage1AssignmentCandidate(
            delta_cost=final_delta,
            placements=placements,
            previews=local_previews,
        )

    def _select_repair_candidate(
        self,
        candidates: List[Stage1AssignmentCandidate],
        group: AllocationGroup,
    ) -> Stage1AssignmentCandidate:
        candidates.sort(key=lambda candidate: self._candidate_sort_key(candidate, group))
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

    def _candidate_sort_key(
        self,
        candidate: Stage1AssignmentCandidate,
        group: AllocationGroup,
    ) -> Tuple[float, int, int, Tuple[Tuple[str, int], ...]]:
        unique_areas = len({placement.area_id for placement in candidate.placements})
        if group.container_count > self._target_containers_per_area(group):
            area_rank = -unique_areas
        else:
            area_rank = len(candidate.placements)
        return (
            candidate.delta_cost,
            area_rank,
            len(candidate.placements),
            tuple((placement.area_id, placement.column_demand) for placement in candidate.placements),
        )

    def _min_required_area_count(self, group: AllocationGroup) -> int:
        if group.container_count <= 100:
            return 1
        target = self._target_containers_per_area(group)
        desired = math.ceil(max(1, group.container_count) / target)
        compatible = max(1, group.column_demand)
        return min(
            self.config.max_split_parts,
            compatible,
            max(2, desired),
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
        workload_overlay_raw: Dict[Tuple[str, int], List[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        physical_cost = 0.0
        infeasible_count = 0

        for group_id, placements in solution.placements_by_group.items():
            group = group_by_id.get(group_id)
            if group is None:
                infeasible_count += 1
                continue
            placement_container_counts = self._placement_container_counts(group, placements)
            group_infeasible = False
            if len(placements) > self.config.max_split_parts:
                group_infeasible = True
            if sum(placement.column_demand for placement in placements) != group.column_demand:
                group_infeasible = True
            for placement, placement_container_count in zip(
                placements,
                placement_container_counts,
            ):
                if placement.column_demand <= 0:
                    group_infeasible = True
                    continue
                area = area_by_id.get(placement.area_id)
                if area is None or not ConstraintChecker.can_assign_to_area(group, area):
                    group_infeasible = True
                    continue
                state = states[placement.area_id]
                preview = state.preview_place(group, placement.column_demand)
                if not preview.feasible:
                    group_infeasible = True
                    continue
                state.apply_preview(preview)
                physical_cost += preview.physical_cost
                area_load[placement.area_id] += placement.column_demand
                line_area_load[(group.line_key, placement.area_id)] += placement.column_demand
                self._accumulate_placement_container_overlay(
                    workload_overlay_raw=workload_overlay_raw,
                    group=group,
                    placement=placement,
                    n_containers=placement_container_count,
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
            workload_overlay={
                key: (value[0], value[1]) for key, value in workload_overlay_raw.items()
            },
            physical_cost=physical_cost,
            infeasible_count=infeasible_count,
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
        if group.column_demand <= 0:
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
                        * placement.column_demand
                        / group.column_demand
                    ),
                )
                count = min(count, remaining)
            counts.append(float(count))
            remaining = max(0, remaining - count)
        return counts

    def _accumulate_placement_container_overlay(
        self,
        *,
        workload_overlay_raw: Dict[Tuple[str, int], List[float]],
        group: AllocationGroup,
        placement: Stage1Placement,
        n_containers: float,
    ) -> None:
        if n_containers <= 0:
            logger.debug(
                "Stage1 workload overlay skipped: group=%s area=%s container_count=%s",
                group.group_id,
                placement.area_id,
                n_containers,
            )
            return

        step_ids = self._voyage_step_span_for_group(group)
        if not step_ids:
            return

        per_step_containers = n_containers / len(step_ids)
        for step_id in step_ids:
            inbound_delta, outbound_delta = self._placement_container_move_delta(
                group,
                per_step_containers,
            )
            key = (placement.area_id, step_id)
            workload_overlay_raw[key][0] += inbound_delta
            workload_overlay_raw[key][1] += outbound_delta

    def _voyage_step_span_for_group(self, group: AllocationGroup) -> List[int]:
        if group.voyage_id in self._voyage_step_span:
            return self._voyage_step_span[group.voyage_id]

        vessel = self._vessels.get(group.voyage_id)
        if vessel is None or not self._time_steps:
            return []

        eta = vessel.eta
        # Fallback keeps grouping's latest departure semantics when ETD is absent,
        # otherwise uses a conservative 48-hour in-yard window.
        etd = vessel.etd or group.latest_departure or eta + timedelta(hours=48)
        if etd <= eta:
            etd = group.latest_departure or eta + timedelta(hours=48)

        span: List[int] = []
        for index, step in enumerate(self._time_steps):
            is_last = index == len(self._time_steps) - 1
            overlaps = step.start_time < etd and step.end_time > eta
            if not overlaps and is_last and eta == step.end_time:
                overlaps = True
            if overlaps:
                span.append(step.step_id)
        self._voyage_step_span[group.voyage_id] = span
        return span

    def _time_step_workload_cost(
        self,
        context: Stage1CostContext,
        yard_areas: List[YardArea],
    ) -> float:
        if self.workload_snapshot is None:
            return 0.0
        by_business: Dict[BusinessType, List[YardArea]] = defaultdict(list)
        for area in yard_areas:
            by_business[area.business_type].append(area)

        total_cost = 0.0
        for step_id, weight in self._iter_relevant_workload_steps():
            if weight <= 0.0:
                continue
            for business_type in (BusinessType.IMPORT, BusinessType.EXPORT):
                areas = by_business.get(business_type, [])
                if not areas:
                    continue
                workloads = [
                    self._area_step_workload(context, area.area_id, step_id)[2]
                    for area in areas
                ]
                if not workloads:
                    continue
                soft_target = self._workload_area_soft_target(business_type)
                overload = sum(
                    max(0.0, workload - soft_target) ** 2 for workload in workloads
                )
                spread = max(workloads) - min(workloads)
                total_cost += weight * self.config.workload_overload_weight * overload
                total_cost += (
                    weight
                    * self.config.workload_balance_weight
                    * self.config.workload_spread_weight_scale
                    * spread
                    * spread
                )
        return total_cost

    def _workload_area_soft_target(self, business_type: BusinessType) -> float:
        if business_type == BusinessType.EXPORT:
            return max(1.0, self.config.workload_area_soft_target_export)
        return max(1.0, self.config.workload_area_soft_target_import)

    def _iter_relevant_workload_steps(self) -> List[Tuple[int, float]]:
        if self.workload_snapshot is None:
            return []
        target_step_ids = set(self.workload_snapshot.target_step_ids.values())
        if self.config.workload_use_target_eta_step_only:
            return [(step_id, 1.0) for step_id in sorted(target_step_ids)]
        return [
            (
                step.step_id,
                1.0
                if step.step_id in target_step_ids
                else self.config.workload_non_target_step_weight,
            )
            for step in self.workload_snapshot.steps
        ]

    def _area_step_workload(
        self,
        context: Stage1CostContext,
        area_id: str,
        step_id: int,
    ) -> Tuple[float, float, float, float]:
        if self.workload_snapshot is None:
            return 0.0, 0.0, 0.0, 0.0
        base = self.workload_snapshot.by_area_step.get((area_id, step_id))
        base_inbound = base.inbound_moves if base is not None else 0.0
        base_outbound = base.outbound_moves if base is not None else 0.0
        max_moves = (
            base.max_moves
            if base is not None
            else self.workload_estimation_config.max_moves_per_step
        )
        overlay_inbound, overlay_outbound = context.workload_overlay.get(
            (area_id, step_id),
            (0.0, 0.0),
        )
        inbound = base_inbound + overlay_inbound
        outbound = base_outbound + overlay_outbound
        return inbound, outbound, inbound + outbound, max_moves

    @staticmethod
    def _placement_container_move_delta(
        group: AllocationGroup,
        n_containers: float,
    ) -> Tuple[float, float]:
        move_delta = float(max(0.0, n_containers))
        if group.business_type == BusinessType.EXPORT:
            return 0.3 * move_delta, 0.7 * move_delta
        return 0.7 * move_delta, 0.3 * move_delta

    def _planning_capacity(self, area_id: str, states: Dict[str, AreaResourceState]) -> int:
        capacity = max(1, states[area_id].initial_total_columns)
        return max(1, int(math.floor(capacity * self.config.workload_soft_capacity_ratio)))

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
                group.column_demand,
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
            if group.column_demand
            <= self._planning_remaining_capacity(area.area_id, context.area_load, context.states)
        ]
        if not capacity_areas:
            split_candidate = self._repair_group_with_split(
                solution,
                group,
                group_by_id,
                yard_areas,
                context,
                self.evaluate(solution, group_by_id, yard_areas),
            )
            if split_candidate is not None:
                return "split feasible but was not selected or accepted"
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

        split_candidate = self._repair_group_with_split(
            solution,
            group,
            group_by_id,
            yard_areas,
            context,
            self.evaluate(solution, group_by_id, yard_areas),
        )
        if split_candidate is not None:
            return "split feasible but was not selected or accepted"

        if preview_failures:
            reason, count = max(preview_failures.items(), key=lambda item: item[1])
            return f"preview infeasible: {reason} ({count} areas)"
        return "split infeasible"

    def _log_workload_balance(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
        yard_areas: List[YardArea],
    ) -> None:
        if self.workload_snapshot is None:
            print("\n【Stage1】无工作量快照，跳过 ETA 时间步打印\n")
            return
        context = self._replay_solution(solution, group_by_id, yard_areas)
        print_eta_step_workloads(
            self.workload_snapshot,
            yard_areas,
            title="【Stage1 分配后】ETA 目标时间步 · 各箱区场桥工作量（箱次；计划箱数按 ETA～ETD 步数均摊）",
            resolve_workload=lambda area_id, step_id: self._area_step_workload(
                context,
                area_id,
                step_id,
            ),
        )

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

    def _group_dispersion_cost(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
    ) -> float:
        cost = 0.0
        min_part = max(1.0, float(self.config.min_split_part_containers))
        for group_id, placements in solution.placements_by_group.items():
            group = group_by_id.get(group_id)
            if group is None or not placements:
                continue
            placement_counts = self._placement_container_counts(group, placements)
            if not placement_counts:
                continue

            target = self._target_containers_per_area(group)
            peak = max(placement_counts)
            peak_excess_ratio = max(0.0, (peak - target) / target)
            cost += self.config.group_peak_weight * peak_excess_ratio ** 2
            if peak > target:
                cost += self.config.group_peak_weight * 0.5 * (
                    (peak - target) / max(1.0, group.container_count)
                )

            if len(placements) > 1:
                for count in placement_counts:
                    fragment_ratio = max(0.0, (min_part - count) / min_part)
                    cost += self.config.fragment_weight * fragment_ratio ** 2
        return cost

    def _split_operation_cost(
        self,
        solution: Stage1Solution,
        group_by_id: Dict[str, AllocationGroup],
    ) -> float:
        cost = 0.0
        for group_id, placements in solution.placements_by_group.items():
            part_count = len(placements)
            if part_count > 1:
                group = group_by_id.get(group_id)
                split_weight = self.config.split_weight
                if group is not None:
                    split_weight = self._effective_split_weight(group)
                cost += split_weight * (part_count - 1) ** 2
        return cost

    def _target_containers_per_area(self, group: AllocationGroup) -> float:
        if group.business_type == BusinessType.EXPORT:
            return max(1.0, float(self.config.export_target_containers_per_area))
        return max(1.0, float(self.config.import_target_containers_per_area))

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
        return (candidate_count, size_rank, -group.column_demand, 0 if group.is_edge_only else 1)

    def _materialize_solution(
        self,
        solution: Stage1Solution,
        groups: List[AllocationGroup],
        area_by_id: Dict[str, YardArea],
    ) -> Tuple[List[AreaAssignment], List[AllocationGroup], List[AllocationGroup]]:
        group_by_id = {group.group_id: group for group in groups}
        assignments: List[AreaAssignment] = []
        unassigned: List[AllocationGroup] = []
        generated_groups: List[AllocationGroup] = []

        for group in groups:
            placements = solution.placements_by_group.get(group.group_id, [])
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
                        column_demand=placement.column_demand,
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
                        column_demand=placement.column_demand,
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
        total_demand = max(1, group.column_demand)
        total_containers = max(group.container_count, len(containers))
        remaining_container_count = total_containers
        split_groups: List[AllocationGroup] = []

        for idx, placement in enumerate(placements, start=1):
            if idx < len(placements):
                count = max(0, round(total_containers * placement.column_demand / total_demand))
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
                column_demand=placement.column_demand,
                earliest_arrival=group.earliest_arrival,
                latest_departure=group.latest_departure,
                is_split=True,
                parent_group_id=group.group_id,
                split_index=idx,
            )
            split_groups.append(sub_group)

        return split_groups


class Stage2BayAllocator:
    """
    Stage 2: within each assigned yard area, decide concrete bay/column usage.

    The implementation delegates to the SCIP model in `stage2_scip` while
    keeping the historical public interface unchanged.
    """

    def __init__(self, scip_config=None):
        self.scip_config = scip_config

    def allocate(
        self,
        area_assignments: List[AreaAssignment],
        groups: Dict[str, AllocationGroup],
        yard_areas: Dict[str, YardArea],
    ) -> List[BayColumnAllocation]:
        from yardplan_core.stage2_scip import ScipStage2BayAllocator

        return ScipStage2BayAllocator(self.scip_config).allocate(
            area_assignments,
            groups,
            yard_areas,
        )


class AllocationEngine:
    """Orchestrates stage 1 + existing stage 2 for Function 2."""

    def __init__(
        self,
        stage1: Optional[Stage1YardAreaAssigner] = None,
        stage2: Optional[Stage2BayAllocator] = None,
    ):
        self.stage1 = stage1 or Stage1YardAreaAssigner()
        self.stage2 = stage2 or Stage2BayAllocator()

    def allocate(
        self,
        groups: List[AllocationGroup],
        yard_areas: List[YardArea],
        workload_snapshot: Optional[AreaWorkloadSnapshot] = None,
    ) -> Tuple[List[AreaAssignment], List[BayColumnAllocation], List[AllocationGroup]]:
        self.stage1.set_workload_snapshot(workload_snapshot)
        area_assignments, unassigned = self.stage1.assign(groups, yard_areas)
        group_dict = {group.group_id: group for group in groups}
        bay_allocations = self.stage2.allocate(
            area_assignments,
            group_dict,
            {area.area_id: area for area in yard_areas},
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
    "AreaScoringStrategy",
    "DefaultAreaScoringStrategy",
    "Stage1YardAreaAssigner",
    "Stage2BayAllocator",
    "AllocationEngine",
]
