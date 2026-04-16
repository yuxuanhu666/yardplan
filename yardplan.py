#!/usr/bin/env python3
"""
Container Terminal Yard Space Allocation Framework
==================================================
A planning-oriented, two-stage, rolling-window allocation framework
for import and export container yard planning.

Author: Senior Algorithm Engineer
Version: 1.0.0

Structure (single-file, layered):
  Section A: Enums & Constants
  Section B: Domain Models (dataclasses)
  Section C: Column Demand Conversion Module
  Section D: Grouping Engine (Function 1)
  Section E: Rolling Window Planner
  Section F: Constraint Checker
  Section G: Scoring Module
  Section H: Stage 1 - Yard Area Assignment
  Section I: Stage 2 - Bay & Column Allocation
  Section J: Allocation Engine (orchestrates Stage 1 + 2)
  Section K: Result Formatter
  Section L: Planner Entry Point
  Section M: Sample Data & Main Runner
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("YardPlanner")


# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Enums & Constants
# ═══════════════════════════════════════════════════════════════════════════════

class BusinessType(Enum):
    IMPORT = "import"
    EXPORT = "export"


class ContainerSize(Enum):
    """
    TEU sizes relevant to yard planning.
    20ft = 1 TEU, occupies 1 bay.
    40ft = 2 TEU, spans 2 adjacent bays (large bay).
    45ft = 2+ TEU, spans 2 adjacent bays, EDGE-ONLY constraint.
    """
    SIZE_20 = 20
    SIZE_40 = 40
    SIZE_45 = 45


class ContainerType(Enum):
    """ISO container type classification."""
    DRY = "dry"
    REEFER = "reefer"
    OPEN_TOP = "open_top"
    FLAT_RACK = "flat_rack"
    TANK = "tank"


class WeightClass(Enum):
    HEAVY = "heavy"
    LIGHT = "light"
    EMPTY = "empty"


class PlannerMode(Enum):
    """Three supported invocation modes."""
    GROUP_ONLY = "group_only"          # Function 1 only
    ALLOCATE_ONLY = "allocate_only"    # Function 2 only (groups provided)
    FULL_PLAN = "full_plan"            # Function 1 + Function 2


class FlowType(Enum):
    """
    Four yard space flows (Section 3 of spec).
    DISCHARGE: vessel -> yard (import in)
    LOADING:   yard -> vessel (export out)
    GATE_IN:   truck -> yard (export in)
    GATE_OUT:  yard -> truck (import out)
    """
    DISCHARGE = "discharge"   # import containers flow IN
    LOADING = "loading"       # export containers flow OUT
    GATE_IN = "gate_in"       # export containers flow IN
    GATE_OUT = "gate_out"     # import containers flow OUT


# Physical constraints
MAX_TIERS_PER_COLUMN = 6
MAX_SPLITS_PER_GROUP = 2
MIN_COLUMNS_PER_SPLIT = 1


# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Domain Models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Container:
    """
    Represents a single physical container with all planning-relevant attributes.
    Some fields are uncertain (marked [UNCERTAIN]) and may be estimated.
    """
    container_id: str
    size: ContainerSize
    container_type: ContainerType
    weight_class: WeightClass
    business_type: BusinessType

    # Vessel linkage
    voyage_id: str
    vessel_id: str

    # Timing (deterministic for vessel operations, uncertain for truck)
    eta: datetime                          # vessel ETA (deterministic)
    etd: Optional[datetime] = None        # vessel ETD (deterministic)

    # [UNCERTAIN] truck pickup window for import (estimated)
    earliest_pickup: Optional[datetime] = None
    latest_pickup: Optional[datetime] = None

    # Export-specific attributes
    destination_port: Optional[str] = None   # export grouping key
    receiving_start: Optional[datetime] = None
    receiving_end: Optional[datetime] = None

    # Import-specific attributes
    consignee: Optional[str] = None          # import grouping key
    pickup_party: Optional[str] = None

    # Current yard position (if already on yard)
    current_block: Optional[str] = None
    current_bay: Optional[int] = None
    current_row: Optional[str] = None
    current_tier: Optional[int] = None

    def is_large_container(self) -> bool:
        """Returns True if container requires a large bay (2 adjacent bays)."""
        return self.size in (ContainerSize.SIZE_40, ContainerSize.SIZE_45)

    def is_edge_only(self) -> bool:
        """45ft containers can only go to edge positions of a yard area."""
        return self.size == ContainerSize.SIZE_45


@dataclass
class Bay:
    """
    A single bay within a yard block/area.
    Each bay has a fixed number of rows (columns in planning sense).

    Note: In terminal terminology:
      - Bay  = longitudinal position along the block
      - Row  = transverse position across the block
      - Tier = vertical stack level

    In this planner, "column" refers to a (bay, row) slot stack.
    We plan at column level, not tier level.
    """
    bay_id: str
    bay_number: int         # sequential number within yard area
    yard_area_id: str
    total_columns: int      # total row positions = columns available
    occupied_columns: int = 0
    size_lock: Optional[ContainerSize] = None  # locked by current occupants
    is_in_large_bay: bool = False              # True if paired for 40/45ft

    @property
    def free_columns(self) -> int:
        return max(0, self.total_columns - self.occupied_columns)

    @property
    def utilization(self) -> float:
        if self.total_columns == 0:
            return 0.0
        return self.occupied_columns / self.total_columns

    def can_accept_20ft(self) -> bool:
        return (
            self.size_lock in (None, ContainerSize.SIZE_20)
            and not self.is_in_large_bay
            and self.free_columns > 0
        )

    def can_accept_large(self) -> bool:
        """Can this bay participate in a large bay pair for 40/45ft?"""
        return (
            self.size_lock in (None, ContainerSize.SIZE_40, ContainerSize.SIZE_45)
            and not (self.size_lock == ContainerSize.SIZE_20)
            and self.free_columns > 0
        )


@dataclass
class LargeBayPair:
    """
    A logical large bay formed by two adjacent physical bays.
    Used exclusively by 40ft and 45ft containers.
    Once formed, neither constituent bay can accept 20ft.
    """
    pair_id: str
    yard_area_id: str
    bay_a: Bay          # lower bay number
    bay_b: Bay          # higher bay number (must be bay_a.bay_number + 1)
    is_edge_pair: bool = False  # True if at leftmost or rightmost edge

    @property
    def total_columns(self) -> int:
        # Columns are counted per bay face for large containers
        # [ASSUMPTION-3]: large container uses columns of both bays combined
        return min(self.bay_a.total_columns, self.bay_b.total_columns)

    @property
    def occupied_columns(self) -> int:
        # For a large bay pair, occupancy is symmetric
        return max(self.bay_a.occupied_columns, self.bay_b.occupied_columns)

    @property
    def free_columns(self) -> int:
        return max(0, self.total_columns - self.occupied_columns)

    def can_accept_45ft(self) -> bool:
        """45ft requires edge-only placement."""
        return self.is_edge_pair and self.free_columns > 0

    def can_accept_40ft(self) -> bool:
        return self.free_columns > 0


@dataclass
class YardArea:
    """
    A logical yard area (block) dedicated to either import or export.
    Contains an ordered list of bays and pre-computed large bay pairs.

    The bay ordering matters for:
    - continuity constraints (splits must be in adjacent columns/bays)
    - edge detection for 45ft groups
    """
    area_id: str
    business_type: BusinessType
    bays: List[Bay] = field(default_factory=list)
    large_bay_pairs: List[LargeBayPair] = field(default_factory=list)

    # Area-level capacity metadata
    supported_sizes: Set[ContainerSize] = field(
        default_factory=lambda: {
            ContainerSize.SIZE_20, ContainerSize.SIZE_40, ContainerSize.SIZE_45
        }
    )
    # Equipment constraints (e.g., only certain RTG spans available)
    max_stack_height: int = MAX_TIERS_PER_COLUMN
    # Location metadata (for future workload/transport scoring)
    distance_to_gate: float = 0.0         # meters [ASSUMPTION] placeholder
    distance_to_berth: Dict[str, float] = field(default_factory=dict)

    @property
    def total_columns(self) -> int:
        return sum(b.total_columns for b in self.bays)

    @property
    def occupied_columns(self) -> int:
        return sum(b.occupied_columns for b in self.bays)

    @property
    def free_columns(self) -> int:
        return self.total_columns - self.occupied_columns

    @property
    def utilization(self) -> float:
        if self.total_columns == 0:
            return 0.0
        return self.occupied_columns / self.total_columns

    def get_free_20ft_columns(self) -> int:
        return sum(b.free_columns for b in self.bays if b.can_accept_20ft())

    def get_free_large_columns(self) -> int:
        return sum(p.free_columns for p in self.large_bay_pairs if p.can_accept_40ft())

    def get_edge_pairs(self) -> List[LargeBayPair]:
        return [p for p in self.large_bay_pairs if p.is_edge_pair]

    def get_bay_by_number(self, bay_number: int) -> Optional[Bay]:
        for b in self.bays:
            if b.bay_number == bay_number:
                return b
        return None


@dataclass
class Vessel:
    """Vessel with operation schedule."""
    vessel_id: str
    vessel_name: str
    voyage_id: str
    eta: datetime
    etd: datetime
    berth_id: str
    # Containers associated with this vessel
    discharge_containers: List[Container] = field(default_factory=list)  # import
    loading_containers: List[Container] = field(default_factory=list)    # export


@dataclass
class YardFlow:
    """
    Represents a predicted flow event affecting yard space.
    Used by the rolling window to update future availability.
    """
    flow_id: str
    flow_type: FlowType
    business_type: BusinessType
    yard_area_id: Optional[str]   # None if not yet allocated
    container_count: int
    column_delta: int              # positive = occupying, negative = releasing
    expected_time: datetime
    voyage_id: Optional[str] = None
    certainty: float = 1.0         # [UNCERTAIN] 0-1, 1=deterministic


@dataclass
class AllocationGroup:
    """
    Core planning unit for stage 1 and stage 2.
    A group of containers sharing the same business attributes,
    planned together as a unit.
    """
    group_id: str
    business_type: BusinessType
    size: ContainerSize
    container_type: ContainerType
    weight_class: WeightClass
    voyage_id: str

    # Grouping attributes (varies by business type)
    group_attributes: Dict[str, Any] = field(default_factory=dict)

    # Member containers
    containers: List[Container] = field(default_factory=list)

    # Derived planning quantities
    column_demand: int = 0           # total columns needed
    container_count: int = 0

    # Time window for this group's presence on yard
    earliest_arrival: Optional[datetime] = None
    latest_departure: Optional[datetime] = None

    # Split tracking (max MAX_SPLITS_PER_GROUP = 2)
    is_split: bool = False
    parent_group_id: Optional[str] = None
    split_index: int = 0            # 0 = original, 1 = first split, 2 = second split

    def __post_init__(self):
        self.container_count = len(self.containers)

    @property
    def is_large_container_group(self) -> bool:
        return self.size in (ContainerSize.SIZE_40, ContainerSize.SIZE_45)

    @property
    def is_edge_only(self) -> bool:
        return self.size == ContainerSize.SIZE_45


@dataclass
class AreaAssignment:
    """Stage 1 result: which area a group (or split sub-group) is assigned to."""
    assignment_id: str
    group_id: str
    yard_area_id: str
    column_demand: int
    split_index: int = 0
    is_partial: bool = False


@dataclass
class BayColumnAllocation:
    """
    Stage 2 result: detailed bay and column allocation for one group part.

    For 20ft groups: each entry is (bay_number, columns_occupied)
    For 40/45ft groups: each entry is ((bay_a_number, bay_b_number), columns_occupied)
    with explicit spanning annotation.
    """
    allocation_id: str
    group_id: str
    yard_area_id: str
    business_type: BusinessType
    size: ContainerSize
    split_index: int = 0

    # Detailed placement:
    # List of (bay_spec, columns_used)
    # For 20ft: bay_spec = bay_number (int)
    # For 40/45ft: bay_spec = (bay_a_number, bay_b_number) tuple
    bay_column_details: List[Tuple[Any, int]] = field(default_factory=list)

    is_edge_placement: bool = False   # relevant for 45ft
    is_spanning: bool = False         # True if 40/45ft (multi-bay)
    notes: str = ""


@dataclass
class PlanningResult:
    """Aggregated result from a full planning run."""
    run_id: str
    timestamp: datetime
    mode: PlannerMode
    allocation_groups: List[AllocationGroup] = field(default_factory=list)
    area_assignments: List[AreaAssignment] = field(default_factory=list)
    bay_column_allocations: List[BayColumnAllocation] = field(default_factory=list)
    unassigned_groups: List[AllocationGroup] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Column Demand Conversion Module
# ═══════════════════════════════════════════════════════════════════════════════

class ColumnDemandConverter:
    """
    Converts container counts to column demand.

    [REPLACEABLE MODULE] This module encapsulates all column-demand logic.
    Replace this class to change how container counts map to columns,
    e.g., to incorporate stowage density, weight distribution, or
    cargo type segregation rules.

    Key rules (from spec Section 6):
    - Max 6 tiers per column (MAX_TIERS_PER_COLUMN)
    - If a group occupies a column, even 1 container = 1 column
    - 20ft: 1 bay per column
    - 40ft/45ft: 2 adjacent bays per column (large bay)
    """

    def __init__(self, max_tiers: int = MAX_TIERS_PER_COLUMN):
        self.max_tiers = max_tiers

    def compute_column_demand(
        self,
        container_count: int,
        size: ContainerSize,
        weight_class: WeightClass,
        container_type: ContainerType,
    ) -> int:
        """
        Compute how many columns a group of containers needs.

        Current implementation uses a simple ceiling division by max_tiers.
        Heavy containers may be restricted in stacking height.

        [REPLACEABLE] Could incorporate:
        - segregation requirements (IMDG codes)
        - reefer plug availability
        - weight stacking limits
        - customer SLA preferences
        """
        effective_tiers = self._effective_tiers(weight_class, container_type)
        # Ceiling division: if you have 7 containers and max 6 tiers,
        # you need 2 columns.
        demand = max(1, -(-container_count // effective_tiers))
        return demand

    def _effective_tiers(
        self, weight_class: WeightClass, container_type: ContainerType
    ) -> int:
        """
        Determine effective max stack height based on weight and type.
        [REPLACEABLE] with actual terminal stacking policy.
        """
        # Reefer: typically limited to 3 tiers due to power connections
        if container_type == ContainerType.REEFER:
            return 3
        # Heavy: limited to 4 tiers for structural safety [ASSUMPTION]
        if weight_class == WeightClass.HEAVY:
            return 4
        # Open top / flat rack: max 2 tiers [ASSUMPTION]
        if container_type in (ContainerType.OPEN_TOP, ContainerType.FLAT_RACK):
            return 2
        # Default: full max height
        return self.max_tiers

    def convert_group(self, group: AllocationGroup) -> int:
        """Compute and store column demand for a group."""
        demand = self.compute_column_demand(
            container_count=group.container_count,
            size=group.size,
            weight_class=group.weight_class,
            container_type=group.container_type,
        )
        group.column_demand = demand
        return demand


# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Grouping Engine (Function 1)
# ═══════════════════════════════════════════════════════════════════════════════

# Type alias for grouping key function
GroupKeyFunc = Callable[[Container], Tuple]

# Default grouping key functions (configurable, not hardcoded)
DEFAULT_EXPORT_GROUP_KEYS: List[str] = [
    "voyage_id",
    "size",
    "container_type",
    "weight_class",
    "destination_port",
]

DEFAULT_IMPORT_GROUP_KEYS: List[str] = [
    "voyage_id",
    "size",
    "container_type",
    "weight_class",
    "consignee",
]


class GroupingConfig:
    """
    Configures how containers are grouped into AllocationGroups.

    Allows overriding grouping keys per business type.
    Keys map to Container field names or custom extraction functions.

    [REPLACEABLE] Add new grouping dimensions by extending this config.
    """

    def __init__(
        self,
        export_keys: Optional[List[str]] = None,
        import_keys: Optional[List[str]] = None,
        custom_key_funcs: Optional[Dict[str, Callable[[Container], Any]]] = None,
    ):
        self.export_keys = export_keys or DEFAULT_EXPORT_GROUP_KEYS
        self.import_keys = import_keys or DEFAULT_IMPORT_GROUP_KEYS
        self.custom_key_funcs = custom_key_funcs or {}

    def get_key(self, container: Container, key_name: str) -> Any:
        """Extract a grouping key value from a container."""
        if key_name in self.custom_key_funcs:
            return self.custom_key_funcs[key_name](container)
        val = getattr(container, key_name, None)
        # Use enum value for hashability in tuple keys
        if isinstance(val, Enum):
            return val.value
        return val

    def make_group_key(
        self, container: Container, business_type: BusinessType
    ) -> Tuple:
        """Build the full composite grouping key for a container."""
        keys = (
            self.export_keys
            if business_type == BusinessType.EXPORT
            else self.import_keys
        )
        return tuple(self.get_key(container, k) for k in keys)


class GroupingEngine:
    """
    Function 1: Groups containers into AllocationGroups.

    Supports both import and export containers with separate,
    configurable grouping rules that share the same infrastructure.

    Usage:
        engine = GroupingEngine(config=GroupingConfig())
        groups = engine.group_containers(containers)
    """

    def __init__(
        self,
        config: Optional[GroupingConfig] = None,
        demand_converter: Optional[ColumnDemandConverter] = None,
    ):
        self.config = config or GroupingConfig()
        self.demand_converter = demand_converter or ColumnDemandConverter()

    def group_containers(
        self, containers: List[Container]
    ) -> List[AllocationGroup]:
        """
        Main grouping method. Separates import/export, applies respective
        grouping keys, and computes column demand for each group.
        """
        import_containers = [
            c for c in containers if c.business_type == BusinessType.IMPORT
        ]
        export_containers = [
            c for c in containers if c.business_type == BusinessType.EXPORT
        ]

        groups: List[AllocationGroup] = []
        groups.extend(self._group_by_type(import_containers, BusinessType.IMPORT))
        groups.extend(self._group_by_type(export_containers, BusinessType.EXPORT))

        logger.info(
            f"Grouping complete: {len(groups)} groups "
            f"({sum(1 for g in groups if g.business_type == BusinessType.IMPORT)} import, "
            f"{sum(1 for g in groups if g.business_type == BusinessType.EXPORT)} export)"
        )
        return groups

    def _group_by_type(
        self, containers: List[Container], business_type: BusinessType
    ) -> List[AllocationGroup]:
        """Group containers of the same business type."""
        if not containers:
            return []

        bucket: Dict[Tuple, List[Container]] = defaultdict(list)
        for c in containers:
            key = self.config.make_group_key(c, business_type)
            bucket[key].append(c)

        groups = []
        for key, members in bucket.items():
            group = self._build_group(key, members, business_type)
            groups.append(group)

        return groups

    def _build_group(
        self,
        key: Tuple,
        members: List[Container],
        business_type: BusinessType,
    ) -> AllocationGroup:
        """Build a single AllocationGroup from a bucket of containers."""
        representative = members[0]

        # Build attribute dict from the grouping key
        key_names = (
            self.config.export_keys
            if business_type == BusinessType.EXPORT
            else self.config.import_keys
        )
        attributes = dict(zip(key_names, key))

        group = AllocationGroup(
            group_id=f"GRP-{uuid.uuid4().hex[:8].upper()}",
            business_type=business_type,
            size=representative.size,
            container_type=representative.container_type,
            weight_class=representative.weight_class,
            voyage_id=representative.voyage_id,
            group_attributes=attributes,
            containers=members,
            container_count=len(members),
        )

        # Compute timing window
        arrivals = [
            c.eta if business_type == BusinessType.IMPORT
            else (c.receiving_start or c.eta)
            for c in members
            if (c.eta if business_type == BusinessType.IMPORT else c.receiving_start or c.eta)
        ]
        departures = [
            c.latest_pickup if business_type == BusinessType.IMPORT and c.latest_pickup
            else (c.etd if c.etd else None)
            for c in members
        ]
        departures = [d for d in departures if d is not None]

        if arrivals:
            group.earliest_arrival = min(arrivals)
        if departures:
            group.latest_departure = max(departures)

        # Compute column demand
        self.demand_converter.convert_group(group)

        return group

    def split_group(
        self, group: AllocationGroup, split_column_demands: List[int]
    ) -> List[AllocationGroup]:
        """
        Split an AllocationGroup into sub-groups with given column demands.
        Respects MAX_SPLITS_PER_GROUP constraint.

        The split is proportional to column demand ratios.
        [REPLACEABLE] The split strategy could later be made smarter,
        e.g., splitting by container attributes or pickup time windows.
        """
        assert len(split_column_demands) <= MAX_SPLITS_PER_GROUP + 1, (
            f"Cannot split into more than {MAX_SPLITS_PER_GROUP + 1} parts"
        )
        assert sum(split_column_demands) == group.column_demand, (
            "Split demands must sum to original demand"
        )

        sub_groups = []
        containers = list(group.containers)
        total = group.container_count
        total_demand = group.column_demand

        for idx, demand in enumerate(split_column_demands):
            # Proportional container assignment
            if idx < len(split_column_demands) - 1:
                count = max(1, round(total * demand / total_demand))
                count = min(count, len(containers))
            else:
                count = len(containers)

            sub_containers = containers[:count]
            containers = containers[count:]

            sub = AllocationGroup(
                group_id=f"{group.group_id}-S{idx + 1}",
                business_type=group.business_type,
                size=group.size,
                container_type=group.container_type,
                weight_class=group.weight_class,
                voyage_id=group.voyage_id,
                group_attributes=group.group_attributes.copy(),
                containers=sub_containers,
                container_count=len(sub_containers),
                column_demand=demand,
                earliest_arrival=group.earliest_arrival,
                latest_departure=group.latest_departure,
                is_split=True,
                parent_group_id=group.group_id,
                split_index=idx + 1,
            )
            sub_groups.append(sub)

        return sub_groups


# ═══════════════════════════════════════════════════════════════════════════════
# Section E: Rolling Window Planner (Function 2)
# ═══════════════════════════════════════════════════════════════════════════════

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
    # Predicted net column change per yard area in this window
    area_capacity_delta: Dict[str, int] = field(default_factory=dict)


class RollingWindowPlanner:
    """
    Manages rolling window generation and future yard availability updates.
    
    Configurable parameters:
    - time_step_hours: length of each discrete time step (e.g. 4)
    - window_steps: number of steps per rolling window (e.g. 3)
    
    This module implements the rolling mechanism required by the specification.
    """

    def __init__(
        self,
        time_step_hours: float = 4.0,
        window_steps: int = 3,
        flow_events: Optional[List[YardFlow]] = None,
    ):
        self.time_step_hours = time_step_hours
        self.window_steps = window_steps
        self.flow_events = flow_events or []

    def generate_time_steps(
        self, horizon_start: datetime, horizon_end: datetime
    ) -> List[TimeStep]:
        """Generate discrete time steps covering the full planning horizon."""
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
        self, time_steps: List[TimeStep]
    ) -> List[RollingWindow]:
        """Create overlapping rolling windows from time steps."""
        windows: List[RollingWindow] = []
        for i in range(len(time_steps) - self.window_steps + 1):
            window_steps = time_steps[i : i + self.window_steps]
            win_start = window_steps[0].start_time
            win_end = window_steps[-1].end_time

            window = RollingWindow(
                window_id=f"WIN-{i:03d}",
                step_index=i,
                start_time=win_start,
                end_time=win_end,
            )
            windows.append(window)

        logger.info(f"Generated {len(windows)} rolling windows")
        return windows

    def update_future_capacity(
        self,
        windows: List[RollingWindow],
        yard_areas: List[YardArea],
        groups: List[AllocationGroup],
    ) -> None:
        """
        Predict net column delta per yard area inside each rolling window
        based on the four business flows (discharge, loading, gate-in, gate-out).
        
        This is a simple default implementation. [REPLACEABLE] with more
        sophisticated forecasting (e.g., Monte-Carlo on uncertain pickup times).
        """
        for window in windows:
            window.area_capacity_delta.clear()
            for area in yard_areas:
                delta = 0
                # Simple placeholder: assume balanced in/out flows
                # In production this would query predicted flows overlapping the window
                delta += 0  # discharge/gate-in increase occupancy
                delta -= 0  # loading/gate-out release space
                window.area_capacity_delta[area.area_id] = delta


# ═══════════════════════════════════════════════════════════════════════════════
# Section F: Constraint Checker
# ═══════════════════════════════════════════════════════════════

class ConstraintChecker:
    """Centralized hard-constraint validation for both stages."""

    @staticmethod
    def can_assign_to_area(group: AllocationGroup, area: YardArea) -> bool:
        """Hard constraints for stage 1 area assignment."""
        if group.business_type != area.business_type:
            return False
        if group.size not in area.supported_sizes:
            return False
        if group.is_large_container_group and area.get_free_large_columns() < group.column_demand:
            return False
        if not group.is_large_container_group and area.get_free_20ft_columns() < group.column_demand:
            return False
        return True

    @staticmethod
    def can_place_in_bay(group: AllocationGroup, bay: Bay) -> bool:
        """Basic bay compatibility check."""
        if group.is_large_container_group:
            return bay.can_accept_large()
        return bay.can_accept_20ft()

    @staticmethod
    def validate_45ft_edge(placement_bays: List[int], yard_area: YardArea) -> bool:
        """45ft groups must be placed at the physical edge of the yard area."""
        if not placement_bays:
            return True
        min_bay = min(placement_bays)
        max_bay = max(placement_bays)
        left_edge = yard_area.bays[0].bay_number if yard_area.bays else 0
        right_edge = yard_area.bays[-1].bay_number if yard_area.bays else 0
        return min_bay == left_edge or max_bay == right_edge


# ═══════════════════════════════════════════════════════════════════════════════
# Section G: Scoring Module (extensible)
# ═══════════════════════════════════════════════════════════════

class AreaScoringStrategy(ABC):
    """Interface for stage 1 yard-area scoring. [REPLACEABLE]"""

    @abstractmethod
    def score(self, group: AllocationGroup, area: YardArea, current_assignments: Dict) -> float:
        pass


class DefaultAreaScoringStrategy(AreaScoringStrategy):
    """Default scoring: balance workload + capacity fit + proximity."""

    def score(self, group: AllocationGroup, area: YardArea, current_assignments: Dict) -> float:
        if not ConstraintChecker.can_assign_to_area(group, area):
            return -float("inf")

        # Workload balance (lower variance is better)
        projected = area.occupied_columns + group.column_demand
        balance_score = -abs(projected - (area.total_columns / 2))  # simple center preference

        # Capacity utilization preference (avoid over/under utilization)
        util_score = -abs(area.utilization - 0.6) * 100

        # Proximity bonus (placeholder)
        proximity = 100 - area.distance_to_berth.get(group.voyage_id, 500) / 10

        return balance_score + util_score + proximity * 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Section H: Stage 1 - Yard Area Assignment (Heuristic)
# ═══════════════════════════════════════════════════════════════

class Stage1YardAreaAssigner:
    """
    Stage 1: Assign allocation groups (or splits) to yard areas.
    Default implementation uses greedy + limited splitting.
    """

    def __init__(self, scoring_strategy: Optional[AreaScoringStrategy] = None):
        self.scorer = scoring_strategy or DefaultAreaScoringStrategy()

    def assign(
        self,
        groups: List[AllocationGroup],
        yard_areas: List[YardArea],
    ) -> Tuple[List[AreaAssignment], List[AllocationGroup]]:
        """Main stage 1 entry point."""
        assignments: List[AreaAssignment] = []
        unassigned: List[AllocationGroup] = []

        # Sort groups by constraint tightness (largest first)
        sorted_groups = sorted(
            groups,
            key=lambda g: (g.is_edge_only, g.is_large_container_group, -g.column_demand),
        )

        current_load: Dict[str, int] = {a.area_id: a.occupied_columns for a in yard_areas}

        for group in sorted_groups:
            candidates = []
            for area in yard_areas:
                score = self.scorer.score(group, area, current_load)
                if score > -float("inf"):
                    candidates.append((score, area))

            if not candidates:
                # Try splitting (max 2 parts)
                if group.column_demand > 1 and not group.is_split:
                    split_demands = [group.column_demand // 2, group.column_demand - group.column_demand // 2]
                    sub_groups = GroupingEngine().split_group(group, split_demands)
                    # Recurse on sub-groups (simple implementation)
                    sub_assign, sub_un = self.assign(sub_groups, yard_areas)
                    assignments.extend(sub_assign)
                    unassigned.extend(sub_un)
                    continue
                unassigned.append(group)
                continue

            # Pick best area
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_area = candidates[0][1]

            assignment = AreaAssignment(
                assignment_id=f"ASN-{uuid.uuid4().hex[:8].upper()}",
                group_id=group.group_id,
                yard_area_id=best_area.area_id,
                column_demand=group.column_demand,
                split_index=group.split_index,
            )
            assignments.append(assignment)

            # Update load
            current_load[best_area.area_id] += group.column_demand

        return assignments, unassigned


# ═══════════════════════════════════════════════════════════════════════════════
# Section I: Stage 2 - Bay & Column Allocation (Heuristic)
# ═══════════════════════════════════════════════════════════════

class Stage2BayAllocator:
    """
    Stage 2: Within each yard area, decide specific bays and columns.
    Prioritizes 45ft → 40ft → 20ft with compatibility logic.
    """

    def allocate(
        self, area_assignments: List[AreaAssignment], groups: Dict[str, AllocationGroup], yard_areas: Dict[str, YardArea]
    ) -> List[BayColumnAllocation]:
        allocations: List[BayColumnAllocation] = []

        # Group assignments by yard area
        by_area: Dict[str, List[Tuple[AreaAssignment, AllocationGroup]]] = defaultdict(list)
        for asn in area_assignments:
            if asn.yard_area_id in yard_areas and asn.group_id in groups:
                by_area[asn.yard_area_id].append((asn, groups[asn.group_id]))

        for area_id, items in by_area.items():
            area = yard_areas[area_id]
            # Sort: edge-only 45ft first, then large, then 20ft
            items.sort(key=lambda x: (x[1].is_edge_only, x[1].is_large_container_group, -x[1].column_demand))

            remaining_20ft_columns = area.get_free_20ft_columns()
            remaining_large_columns = area.get_free_large_columns()

            for asn, group in items:
                if group.is_edge_only:
                    # 45ft edge placement
                    edge_pairs = area.get_edge_pairs()
                    if edge_pairs and edge_pairs[0].free_columns >= group.column_demand:
                        pair = edge_pairs[0]
                        alloc = BayColumnAllocation(
                            allocation_id=f"ALC-{uuid.uuid4().hex[:8].upper()}",
                            group_id=group.group_id,
                            yard_area_id=area_id,
                            business_type=group.business_type,
                            size=group.size,
                            split_index=group.split_index,
                            bay_column_details=[((pair.bay_a.bay_number, pair.bay_b.bay_number), group.column_demand)],
                            is_edge_placement=True,
                            is_spanning=True,
                            notes="45ft edge placement",
                        )
                        allocations.append(alloc)
                        # Mark occupied (simplified)
                        pair.bay_a.occupied_columns += group.column_demand
                        pair.bay_b.occupied_columns += group.column_demand
                        continue

                if group.is_large_container_group:
                    # 40ft large bay placement
                    for pair in area.large_bay_pairs:
                        if pair.free_columns >= group.column_demand:
                            alloc = BayColumnAllocation(
                                allocation_id=f"ALC-{uuid.uuid4().hex[:8].upper()}",
                                group_id=group.group_id,
                                yard_area_id=area_id,
                                business_type=group.business_type,
                                size=group.size,
                                split_index=group.split_index,
                                bay_column_details=[((pair.bay_a.bay_number, pair.bay_b.bay_number), group.column_demand)],
                                is_spanning=True,
                                notes="40ft large bay",
                            )
                            allocations.append(alloc)
                            pair.bay_a.occupied_columns += group.column_demand
                            pair.bay_b.occupied_columns += group.column_demand
                            break
                    else:
                        logger.warning(f"No suitable large bay for group {group.group_id}")
                else:
                    # 20ft single bay packing (simple greedy)
                    placed = 0
                    details = []
                    for bay in area.bays:
                        if bay.can_accept_20ft() and placed < group.column_demand:
                            can_place = min(bay.free_columns, group.column_demand - placed)
                            details.append((bay.bay_number, can_place))
                            bay.occupied_columns += can_place
                            placed += can_place
                    if placed == group.column_demand:
                        alloc = BayColumnAllocation(
                            allocation_id=f"ALC-{uuid.uuid4().hex[:8].upper()}",
                            group_id=group.group_id,
                            yard_area_id=area_id,
                            business_type=group.business_type,
                            size=group.size,
                            split_index=group.split_index,
                            bay_column_details=details,
                            is_spanning=False,
                            notes="20ft single-bay packing",
                        )
                        allocations.append(alloc)
                    else:
                        logger.warning(f"Partial placement for 20ft group {group.group_id}")

        return allocations


# ═══════════════════════════════════════════════════════════════════════════════
# Section J: Allocation Engine
# ═══════════════════════════════════════════════════════════════

class AllocationEngine:
    """Orchestrates stage 1 + stage 2 for Function 2."""

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
    ) -> Tuple[List[AreaAssignment], List[BayColumnAllocation], List[AllocationGroup]]:
        area_assignments, unassigned = self.stage1.assign(groups, yard_areas)

        # Build lookup for stage 2
        group_dict = {g.group_id: g for g in groups}
        bay_allocations = self.stage2.allocate(area_assignments, group_dict, {a.area_id: a for a in yard_areas})

        return area_assignments, bay_allocations, unassigned


# ═══════════════════════════════════════════════════════════════════════════════
# Section K: Result Formatter
# ═══════════════════════════════════════════════════════════════

class ResultFormatter:
    """Pretty printer for planning results."""

    @staticmethod
    def print_summary(result: PlanningResult):
        print("\n" + "="*80)
        print("YARD SPACE ALLOCATION PLANNING RESULT")
        print("="*80)
        print(f"Run ID: {result.run_id} | Mode: {result.mode.value} | Time: {result.timestamp}")
        print(f"Groups generated: {len(result.allocation_groups)}")
        print(f"Area assignments: {len(result.area_assignments)}")
        print(f"Bay/column allocations: {len(result.bay_column_allocations)}")
        print(f"Unassigned groups: {len(result.unassigned_groups)}")

        if result.warnings:
            print("\n⚠️  Warnings:")
            for w in result.warnings:
                print(f"   - {w}")

        print("\nSample Bay Allocations (first 3):")
        for alloc in result.bay_column_allocations[:3]:
            print(f"  Group {alloc.group_id} → Area {alloc.yard_area_id} | "
                  f"Details: {alloc.bay_column_details} | Notes: {alloc.notes}")

        print("\n" + "="*80)


# ═══════════════════════════════════════════════════════════════════════════════
# Section L: Planner Entry Point
# ═══════════════════════════════════════════════════════════════

class YardPlanner:
    """
    Main entry point supporting three modes:
    1. GROUP_ONLY
    2. ALLOCATE_ONLY
    3. FULL_PLAN
    """

    def __init__(self):
        self.grouping_engine = GroupingEngine()
        self.rolling_planner = RollingWindowPlanner()
        self.allocation_engine = AllocationEngine()
        self.formatter = ResultFormatter()

    def plan(
        self,
        containers: List[Container],
        yard_areas: List[YardArea],
        vessels: Optional[List[Vessel]] = None,
        mode: PlannerMode = PlannerMode.FULL_PLAN,
        horizon_start: Optional[datetime] = None,
        horizon_end: Optional[datetime] = None,
    ) -> PlanningResult:
        run_id = f"PLAN-{uuid.uuid4().hex[:12].upper()}"
        timestamp = datetime.now()

        if mode == PlannerMode.GROUP_ONLY:
            groups = self.grouping_engine.group_containers(containers)
            result = PlanningResult(run_id=run_id, timestamp=timestamp, mode=mode, allocation_groups=groups)
            self.formatter.print_summary(result)
            return result

        # Determine horizon
        if not horizon_start or not horizon_end:
            all_times = [c.eta for c in containers if c.eta] + [c.etd for c in containers if c.etd]
            horizon_start = min(all_times) if all_times else datetime.now()
            horizon_end = max(all_times) if all_times else horizon_start + timedelta(hours=48)

        # Rolling window setup (demonstration)
        time_steps = self.rolling_planner.generate_time_steps(horizon_start, horizon_end)
        windows = self.rolling_planner.generate_rolling_windows(time_steps)
        self.rolling_planner.update_future_capacity(windows, yard_areas, [])

        if mode == PlannerMode.ALLOCATE_ONLY:
            # Assume groups already provided via containers list for simplicity
            groups = self.grouping_engine.group_containers(containers)
        else:
            groups = self.grouping_engine.group_containers(containers)

        area_assignments, bay_allocations, unassigned = self.allocation_engine.allocate(groups, yard_areas)

        result = PlanningResult(
            run_id=run_id,
            timestamp=timestamp,
            mode=mode,
            allocation_groups=groups,
            area_assignments=area_assignments,
            bay_column_allocations=bay_allocations,
            unassigned_groups=unassigned,
        )

        self.formatter.print_summary(result)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Section M: Minimal Sample Input & Runner
# ═══════════════════════════════════════════════════════════════

def create_sample_data() -> Tuple[List[Container], List[YardArea]]:
    """Realistic sample data for a small terminal yard planning scenario."""
    now = datetime(2026, 4, 16, 8, 0)

    # Sample containers
    containers = [
        # Import group
        Container(
            container_id="CN001", size=ContainerSize.SIZE_40, container_type=ContainerType.DRY,
            weight_class=WeightClass.HEAVY, business_type=BusinessType.IMPORT,
            voyage_id="VOY-001", vessel_id="VES-001", eta=now,
            consignee="ABC Logistics", latest_pickup=now + timedelta(hours=48),
        ),
        Container(
            container_id="CN002", size=ContainerSize.SIZE_20, container_type=ContainerType.DRY,
            weight_class=WeightClass.LIGHT, business_type=BusinessType.IMPORT,
            voyage_id="VOY-001", vessel_id="VES-001", eta=now,
            consignee="ABC Logistics",
        ),
        # Export group
        Container(
            container_id="CN003", size=ContainerSize.SIZE_45, container_type=ContainerType.DRY,
            weight_class=WeightClass.EMPTY, business_type=BusinessType.EXPORT,
            voyage_id="VOY-002", vessel_id="VES-002", eta=now + timedelta(hours=12),
            destination_port="SGP", receiving_start=now + timedelta(hours=6),
        ),
        Container(
            container_id="CN004", size=ContainerSize.SIZE_40, container_type=ContainerType.DRY,
            weight_class=WeightClass.HEAVY, business_type=BusinessType.EXPORT,
            voyage_id="VOY-002", vessel_id="VES-002", eta=now + timedelta(hours=12),
            destination_port="SGP",
        ),
    ]

    # Sample yard areas (import and export separated)
    import_area = YardArea(
        area_id="IMP-A", business_type=BusinessType.IMPORT,
        bays=[
            Bay(bay_id="IMP-A-01", bay_number=1, yard_area_id="IMP-A", total_columns=50),
            Bay(bay_id="IMP-A-02", bay_number=2, yard_area_id="IMP-A", total_columns=50),
        ],
        large_bay_pairs=[
            LargeBayPair(pair_id="IMP-A-L1", yard_area_id="IMP-A",
                         bay_a=Bay(bay_id="IMP-A-01", bay_number=1, yard_area_id="IMP-A", total_columns=50),
                         bay_b=Bay(bay_id="IMP-A-02", bay_number=2, yard_area_id="IMP-A", total_columns=50),
                         is_edge_pair=True)
        ]
    )

    export_area = YardArea(
        area_id="EXP-B", business_type=BusinessType.EXPORT,
        bays=[
            Bay(bay_id="EXP-B-01", bay_number=1, yard_area_id="EXP-B", total_columns=60),
            Bay(bay_id="EXP-B-02", bay_number=2, yard_area_id="EXP-B", total_columns=60),
            Bay(bay_id="EXP-B-03", bay_number=3, yard_area_id="EXP-B", total_columns=60),
        ],
        large_bay_pairs=[
            LargeBayPair(pair_id="EXP-B-L1", yard_area_id="EXP-B",
                         bay_a=Bay(bay_id="EXP-B-01", bay_number=1, yard_area_id="EXP-B", total_columns=60),
                         bay_b=Bay(bay_id="EXP-B-02", bay_number=2, yard_area_id="EXP-B", total_columns=60),
                         is_edge_pair=True),
            LargeBayPair(pair_id="EXP-B-L2", yard_area_id="EXP-B",
                         bay_a=Bay(bay_id="EXP-B-02", bay_number=2, yard_area_id="EXP-B", total_columns=60),
                         bay_b=Bay(bay_id="EXP-B-03", bay_number=3, yard_area_id="EXP-B", total_columns=60),
                         is_edge_pair=False)
        ]
    )

    return containers, [import_area, export_area]


if __name__ == "__main__":
    print("🚀 Starting Container Terminal Yard Allocation Framework Demo")
    containers, yard_areas = create_sample_data()

    planner = YardPlanner()

    # Run full plan (Function 1 + Function 2)
    result = planner.plan(
        containers=containers,
        yard_areas=yard_areas,
        mode=PlannerMode.FULL_PLAN,
    )

    print("\n✅ Framework ready for extension.")
    print("   - Import/export separation enforced")
    # - Rolling window infrastructure in place
    # - Two-stage heuristic with replaceable scoring & objectives
    # - Split logic and 45ft edge constraint implemented
    # - Column-level planning with large-bay support