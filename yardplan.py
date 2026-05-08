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
  Section M: Main Runner
  Section O: TOS Data Loader
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

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

    # Visit and line linkage
    voyage_id: str
    line_key: Optional[int]
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

    # Raw TOS attributes used to build the range-level response filter.
    iso_type: Optional[str] = None
    category: Optional[int] = None
    pod: Optional[str] = None
    cattier_kind: Optional[str] = None
    trade_code: Optional[str] = None
    freight_kind: Optional[int] = None
    owner_company: Optional[str] = None
    line_company: Optional[str] = None
    truck_company: Optional[str] = None
    belonger_company: Optional[str] = None
    work_type: Optional[int] = None
    bol: Optional[str] = None
    damage_code: Optional[str] = None
    raw_weight: Optional[float] = None
    is_reefer: bool = False
    is_hazardous: bool = False
    is_damage: bool = False
    is_high: bool = False
    is_gauge: bool = False
    is_dirty: bool = False

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
    line_key: Optional[int]

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
    "line_key",
    "size",
    "container_type",
    "weight_class",
    "destination_port",
]

DEFAULT_IMPORT_GROUP_KEYS: List[str] = [
    "line_key",
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
            line_key=representative.line_key,
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
                line_key=group.line_key,
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
# Section N: YardSpace Integration Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class YardSpaceAdapter:
    """
    将 YardSpace (槽位级别的实时状态) 转换为规划引擎所需的 YardArea/Bay 对象。

    映射规则
    --------
    YardSpace                          →  yardplan
    ─────────────────────────────────────────────────────────
    blockId                            →  YardArea.area_id
    奇数 bayIdx 的 stack 列             →  Bay (is_in_large_bay=False)
    偶数 bayIdx 的 stack 列             →  LargeBayPair
    next_placeable_tier is not None    →  free column (可再放箱的 stack)
    next_placeable_tier is None        →  occupied column (已满 stack)

    使用示例
    --------
        from useable_space import YardSpace
        yard = YardSpace.load()
        btypes = {"B01": BusinessType.IMPORT, "B02": BusinessType.EXPORT}
        planner = YardPlanner()
        result = planner.plan_with_yard_space(
            yard, containers, btypes, apply_to_yard=True
        )
    """

    @staticmethod
    def build_yard_areas(
        yard: Any,
        block_business_types: Dict[str, BusinessType],
        block_ids: Optional[List[str]] = None,
    ) -> Tuple[List[YardArea], Dict[str, list]]:
        """
        从 YardSpace 实例构建 YardArea 列表。

        Parameters
        ----------
        yard                 : YardSpace 实例
        block_business_types : blockId -> BusinessType 映射 (进/出口属性由外部配置提供)
        block_ids            : 仅处理指定 block, None 表示全部

        Returns
        -------
        (yard_areas, slot_registry)
            slot_registry: bay_id/pair_id -> [(stack_key, stack_info), ...]
            供后续精细化槽位分配时使用
        """
        # 按 blockId 分组 stacks
        blocks: Dict[str, dict] = defaultdict(dict)
        for (bid, bay_idx, stk_idx), sinfo in yard.stacks.items():
            blocks[bid][(bay_idx, stk_idx)] = sinfo

        slot_registry: Dict[str, list] = {}
        yard_areas: List[YardArea] = []

        target_blocks = block_ids if block_ids else sorted(blocks.keys())

        for block_id in target_blocks:
            if block_id not in blocks:
                continue
            btype = block_business_types.get(block_id, BusinessType.IMPORT)
            stack_dict = blocks[block_id]

            # 按 bayIdx 分组
            bay_groups: Dict[int, list] = defaultdict(list)
            for (bay_idx, stk_idx), sinfo in stack_dict.items():
                bay_groups[bay_idx].append(((block_id, bay_idx, stk_idx), sinfo))

            sorted_bay_idxs = sorted(bay_groups.keys())

            bays: List[Bay] = []
            large_bay_pairs: List[LargeBayPair] = []

            for bay_idx in sorted_bay_idxs:
                stacks_in_bay = bay_groups[bay_idx]
                total = len(stacks_in_bay)
                # free_columns: 该 bayIdx 下 next_placeable_tier 不为 None 的 stack 数量
                free = sum(
                    1 for _, sinfo in stacks_in_bay
                    if sinfo.get("next_placeable_tier") is not None
                )
                occupied = total - free
                is_edge = (
                    bay_idx == sorted_bay_idxs[0] or bay_idx == sorted_bay_idxs[-1]
                )

                if bay_idx % 2 == 1:
                    # ── 奇数 bayIdx → 20ft Bay ──────────────────────────────
                    bay_obj = Bay(
                        bay_id=f"{block_id}-{bay_idx}",
                        bay_number=bay_idx,
                        yard_area_id=block_id,
                        total_columns=total,
                        occupied_columns=occupied,
                        is_in_large_bay=False,
                    )
                    bays.append(bay_obj)
                    slot_registry[bay_obj.bay_id] = stacks_in_bay

                else:
                    # ── 偶数 bayIdx → 40ft LargeBayPair ─────────────────────
                    # 用两个同值 Bay 表示大贝位的对称结构:
                    #   LargeBayPair.total_columns   = min(a.total, b.total)  = total
                    #   LargeBayPair.occupied_columns = max(a.occ, b.occ)    = occupied
                    #   LargeBayPair.free_columns     = total - occupied
                    bay_a = Bay(
                        bay_id=f"{block_id}-{bay_idx}-A",
                        bay_number=bay_idx,
                        yard_area_id=block_id,
                        total_columns=total,
                        occupied_columns=occupied,
                        is_in_large_bay=True,
                    )
                    bay_b = Bay(
                        bay_id=f"{block_id}-{bay_idx}-B",
                        bay_number=bay_idx + 1,
                        yard_area_id=block_id,
                        total_columns=total,
                        occupied_columns=occupied,
                        is_in_large_bay=True,
                    )
                    pair = LargeBayPair(
                        pair_id=f"{block_id}-40-{bay_idx}",
                        yard_area_id=block_id,
                        bay_a=bay_a,
                        bay_b=bay_b,
                        is_edge_pair=is_edge,
                    )
                    large_bay_pairs.append(pair)
                    slot_registry[pair.pair_id] = stacks_in_bay

            area = YardArea(
                area_id=block_id,
                business_type=btype,
                bays=bays,
                large_bay_pairs=large_bay_pairs,
                max_stack_height=MAX_TIERS_PER_COLUMN,
            )
            yard_areas.append(area)

        logger.info(
            f"YardSpaceAdapter: 构建了 {len(yard_areas)} 个 YardArea "
            f"(20ft Bay: {sum(len(a.bays) for a in yard_areas)}, "
            f"40ft LargeBayPair: {sum(len(a.large_bay_pairs) for a in yard_areas)})"
        )
        return yard_areas, slot_registry

    @staticmethod
    def apply_allocation(
        result: "PlanningResult",
        yard: Any,
    ) -> Dict[str, str]:
        """
        将规划结果映射到 YardSpace 中的具体槽位并提交占位。

        对每个已分配 AllocationGroup:
          - 20ft 箱 → 调用 yard.get_placeable_20ft(block_id) 取槽
          - 40/45ft 箱 → 调用 yard.get_placeable_40ft(block_id) 取槽
          - 调用 yard.place_container() 实时写入状态

        每次放箱后立即重新查询可用位, 确保后续分配基于最新状态。

        Parameters
        ----------
        result : PlanningResult (须含 allocation_groups + bay_column_allocations)
        yard   : YardSpace 实例

        Returns
        -------
        {container_id -> assigned_full_slot_name}
        """
        group_by_id: Dict[str, AllocationGroup] = {
            g.group_id: g for g in result.allocation_groups
        }
        assignment_map: Dict[str, str] = {}

        for alloc in result.bay_column_allocations:
            group = group_by_id.get(alloc.group_id)
            if group is None:
                continue
            block_id = alloc.yard_area_id

            is_large = alloc.size in (ContainerSize.SIZE_40, ContainerSize.SIZE_45)
            size_int = {
                ContainerSize.SIZE_20: 1,
                ContainerSize.SIZE_40: 2,
                ContainerSize.SIZE_45: 3,
            }.get(alloc.size, 1)

            for container in group.containers:
                # 每次放箱后重新查询, 保证状态最新
                pool = (
                    yard.get_placeable_40ft(block_id=block_id)
                    if is_large
                    else yard.get_placeable_20ft(block_id=block_id)
                )
                if not pool:
                    logger.warning(
                        f"Block {block_id} 无可用槽位, 容器 {container.container_id} 未能分配"
                    )
                    break
                slot_name = pool[0]["fullSlotName"]
                yard.place_container(slot_name, container.container_id, size_int)
                assignment_map[container.container_id] = slot_name

        logger.info(
            f"YardSpaceAdapter.apply_allocation: 共分配 {len(assignment_map)} 个容器"
        )
        return assignment_map


# ═══════════════════════════════════════════════════════════════════════════════
# Section K: Result Formatter
# ═══════════════════════════════════════════════════════════════

class ResultFormatter:
    """Pretty printer for planning results."""

    @staticmethod
    def build_range_plan(result: PlanningResult) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert group-level bay/column allocations to the API range format.

        The output intentionally stops at block/bay/stack/tier ranges. It does
        not assign individual containers to concrete slots.
        """
        group_by_id: Dict[str, AllocationGroup] = {
            g.group_id: g for g in result.allocation_groups
        }
        data: List[Dict[str, Any]] = []

        for idx, alloc in enumerate(result.bay_column_allocations, start=1):
            group = group_by_id.get(alloc.group_id)
            if group is None:
                continue

            range_list = [
                ResultFormatter._build_range_item(alloc.yard_area_id, bay_spec, columns_used)
                for bay_spec, columns_used in alloc.bay_column_details
            ]
            if not range_list:
                continue

            data.append({
                "groupKey": idx,
                "groupId": idx,
                "rangeList": range_list,
                "filter": ResultFormatter._build_filter(group),
            })

        return {"data": data}

    @staticmethod
    def _build_range_item(
        block_id: str,
        bay_spec: Any,
        columns_used: int,
    ) -> Dict[str, Any]:
        if isinstance(bay_spec, tuple):
            start_bay = min(int(bay_spec[0]), int(bay_spec[1]))
            end_bay = max(int(bay_spec[0]), int(bay_spec[1]))
        else:
            start_bay = end_bay = int(bay_spec)

        return {
            "blockId": block_id,
            "startBayIndex": start_bay,
            "endBayIndex": end_bay,
            "startStackIndex": 1,
            "endStackIndex": max(1, int(columns_used)),
            "startTierIndex": 1,
            "endTierIndex": MAX_TIERS_PER_COLUMN,
        }

    @staticmethod
    def _build_filter(group: AllocationGroup) -> Dict[str, Any]:
        external_filter = group.group_attributes.get("filter")
        if isinstance(external_filter, dict):
            return external_filter

        containers = group.containers

        def unique(attr: str) -> List[Any]:
            values = []
            for container in containers:
                value = getattr(container, attr, None)
                if value is None or value == "":
                    continue
                candidates = value if isinstance(value, list) else [value]
                for candidate in candidates:
                    if candidate is None or candidate == "":
                        continue
                    if candidate not in values:
                        values.append(candidate)
            return values

        weights = [
            ResultFormatter._normalise_weight(c.raw_weight)
            for c in containers
            if c.raw_weight is not None
        ]

        weight_class = {
            WeightClass.EMPTY: 0,
            WeightClass.LIGHT: 1,
            WeightClass.HEAVY: 2,
        }.get(group.weight_class)

        return {
            "filterName": ResultFormatter._filter_name(group),
            "isoType": unique("iso_type"),
            "category": unique("category"),
            "pod": unique("pod"),
            "cattierKind": unique("cattier_kind"),
            "tradeCode": unique("trade_code"),
            "freightKind": unique("freight_kind"),
            "bReefer": any(c.is_reefer for c in containers),
            "bHazardous": any(c.is_hazardous for c in containers),
            "bDamage": any(c.is_damage for c in containers),
            "bHigh": any(c.is_high for c in containers),
            "bGauge": any(c.is_gauge for c in containers),
            "ownerCompany": unique("owner_company"),
            "lineCompany": unique("line_company"),
            "truckCompany": unique("truck_company"),
            "belongerCompany": unique("belonger_company"),
            "bDirty": any(c.is_dirty for c in containers),
            "weightClass": weight_class,
            "weightMin": min(weights) if weights else None,
            "weightMax": max(weights) if weights else None,
            "workType": ResultFormatter._single_or_none(unique("work_type")),
            "bol": unique("bol"),
            "damageCode": unique("damage_code"),
        }

    @staticmethod
    def _filter_name(group: AllocationGroup) -> str:
        business_name = "进口" if group.business_type == BusinessType.IMPORT else "出口"
        weight_name = {
            WeightClass.EMPTY: "空箱",
            WeightClass.LIGHT: "轻箱",
            WeightClass.HEAVY: "重箱",
        }.get(group.weight_class, "箱")
        return f"{business_name}{weight_name}过滤"

    @staticmethod
    def _normalise_weight(weight: Optional[float]) -> Optional[float]:
        if weight is None:
            return None
        value = float(weight)
        if abs(value) > 1000:
            value = value / 1000.0
        return round(value, 3)

    @staticmethod
    def _single_or_none(values: List[Any]) -> Optional[Any]:
        return values[0] if values else None

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
            result.metrics["range_plan"] = {"data": []}
            result.metrics["data"] = []
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
        range_plan = self.formatter.build_range_plan(result)
        result.metrics["range_plan"] = range_plan
        result.metrics["data"] = range_plan["data"]

        self.formatter.print_summary(result)
        return result

    def plan_groups(
        self,
        groups: List[AllocationGroup],
        yard_areas: List[YardArea],
        mode: PlannerMode = PlannerMode.FULL_PLAN,
        horizon_start: Optional[datetime] = None,
        horizon_end: Optional[datetime] = None,
    ) -> PlanningResult:
        """
        Plan pre-built allocation groups without regrouping containers.

        This is used by input type=2, where allocation groups come from an
        external interface and should go directly into the two-stage algorithm.
        """
        run_id = f"PLAN-{uuid.uuid4().hex[:12].upper()}"
        timestamp = datetime.now()

        self._ensure_group_demands(groups)

        if not horizon_start or not horizon_end:
            all_times: List[datetime] = []
            for group in groups:
                if group.earliest_arrival:
                    all_times.append(group.earliest_arrival)
                if group.latest_departure:
                    all_times.append(group.latest_departure)
            horizon_start = min(all_times) if all_times else datetime.now()
            horizon_end = max(all_times) if all_times else horizon_start + timedelta(hours=48)

        time_steps = self.rolling_planner.generate_time_steps(horizon_start, horizon_end)
        windows = self.rolling_planner.generate_rolling_windows(time_steps)
        self.rolling_planner.update_future_capacity(windows, yard_areas, [])

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
        range_plan = self.formatter.build_range_plan(result)
        result.metrics["range_plan"] = range_plan
        result.metrics["data"] = range_plan["data"]

        self.formatter.print_summary(result)
        return result

    def plan_groups_with_yard_space(
        self,
        yard: Any,
        groups: List[AllocationGroup],
        block_business_types: Dict[str, BusinessType],
        block_ids: Optional[List[str]] = None,
        mode: PlannerMode = PlannerMode.FULL_PLAN,
        apply_to_yard: bool = False,
        horizon_start: Optional[datetime] = None,
        horizon_end: Optional[datetime] = None,
    ) -> PlanningResult:
        yard_areas, _slot_registry = YardSpaceAdapter.build_yard_areas(
            yard, block_business_types, block_ids
        )
        result = self.plan_groups(
            groups=groups,
            yard_areas=yard_areas,
            mode=mode,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
        )
        if apply_to_yard:
            assignments = YardSpaceAdapter.apply_allocation(result, yard)
            result.metrics["slot_assignments"] = assignments
            logger.info(
                f"已将规划结果写回 YardSpace: {len(assignments)} 个容器获得槽位"
            )
        return result

    def _ensure_group_demands(self, groups: List[AllocationGroup]) -> None:
        for group in groups:
            if group.column_demand > 0:
                continue
            if group.containers:
                self.grouping_engine.demand_converter.convert_group(group)
                continue
            raise ValueError(
                f"外部分配组 {group.group_id} 缺少 column_demand，"
                "且没有 containers 可用于推算列需求"
            )

    def plan_with_yard_space(
        self,
        yard: Any,
        containers: List[Container],
        block_business_types: Dict[str, BusinessType],
        block_ids: Optional[List[str]] = None,
        mode: PlannerMode = PlannerMode.FULL_PLAN,
        apply_to_yard: bool = False,
        horizon_start: Optional[datetime] = None,
        horizon_end: Optional[datetime] = None,
    ) -> PlanningResult:
        """
        使用 YardSpace 提供的实际堆场状态数据进行规划。

        取代原有的手动构造 YardArea 方式, 自动从 YardSpace 读取当前占位情况,
        计算每个 block 的可用列容量, 并驱动两阶段规划引擎。

        Parameters
        ----------
        yard                 : YardSpace 实例 (来自 useable_space.YardSpace.load())
        containers           : 待规划的 Container 列表
        block_business_types : {blockId: BusinessType} — 各 block 的进/出口属性
                               (YardSpace 中不含此信息, 须由外部配置提供)
        block_ids            : 仅使用指定 block 参与规划, None 表示全部
        mode                 : PlannerMode (GROUP_ONLY / ALLOCATE_ONLY / FULL_PLAN)
        apply_to_yard        : True → 兼容旧流程, 规划完成后额外写回具体槽位
        horizon_start/end    : 规划时间窗口 (可选, 不提供则自动从 containers 推断)

        Returns
        -------
        PlanningResult
            result.metrics["range_plan"] 包含 {"data": [...]} 范围级分配结果。
            apply_to_yard=True 时额外包含 result.metrics["slot_assignments"]。
        """
        yard_areas, _slot_registry = YardSpaceAdapter.build_yard_areas(
            yard, block_business_types, block_ids
        )
        result = self.plan(
            containers=containers,
            yard_areas=yard_areas,
            mode=mode,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
        )
        if apply_to_yard:
            assignments = YardSpaceAdapter.apply_allocation(result, yard)
            result.metrics["slot_assignments"] = assignments
            logger.info(
                f"已将规划结果写回 YardSpace: {len(assignments)} 个容器获得槽位"
            )
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Section O: TOS Data Loader
# ═══════════════════════════════════════════════════════════════════════════════

class TOSLoader:
    """
    从 TOS 接口下载的 JSON 文件中加载规划所需数据。

    数据来源
    --------
    - 217getVesselVisit（船舶访问计划）.json  : 船舶艘次信息 (eta/etd 等)
    - 217getBoundList（装船箱和卸船箱列表）.json : 箱子与艘次的绑定信息

    卸船箱（进口箱）过滤条件（取自 Inbound 列表）
    ----------------------------------------
    - serviceLineKey in line_keys : 匹配目标航线
    - boundType == 1              : 进入堆场方向
    - visitType == 1              : 来自船舶（非进闸）
    """

    # containerISO 后缀 → ContainerType
    _ISO_TYPE_MAP: Dict[str, ContainerType] = {
        "RF": ContainerType.REEFER,
        "OT": ContainerType.OPEN_TOP,
        "PL": ContainerType.FLAT_RACK,
        "FR": ContainerType.FLAT_RACK,
        "TK": ContainerType.TANK,
    }

    # containerSize (int from TOS) → ContainerSize enum
    _SIZE_MAP: Dict[int, ContainerSize] = {
        1: ContainerSize.SIZE_20,
        2: ContainerSize.SIZE_40,
        3: ContainerSize.SIZE_45,
    }

    # 重箱判定阈值 (kg), 超过此值视为重箱
    HEAVY_WEIGHT_THRESHOLD: float = 20000.0

    def __init__(
        self,
        vessel_visit_path: Optional[str] = None,
        bound_list_path: Optional[str] = None,
    ):
        self.vessel_visit_path = vessel_visit_path or os.path.join(
            _DATA_DIR, "217getVesselVisit（船舶访问计划）.json"
        )
        self.bound_list_path = bound_list_path or os.path.join(
            _DATA_DIR, "217getBoundList（装船箱和卸船箱列表）.json"
        )

    # ------------------------------------------------------------------ load

    def load_vessels(self, line_keys: List[int]) -> Dict[str, Vessel]:
        """
        从 VesselVisit JSON 加载目标 lineKey 对应的船舶信息。

        Parameters
        ----------
        line_keys : 目标航线号列表 (对应 lineKey 字段，int)

        Returns
        -------
        {vesselVisitId -> Vessel}
        """
        with open(self.vessel_visit_path, "r", encoding="utf-8") as f:
            raw: List[dict] = json.load(f)

        target_set = {
            lk for lk in (self._coerce_line_key(value) for value in line_keys)
            if lk is not None
        }
        vessels: Dict[str, Vessel] = {}
        matched_line_keys: Set[int] = set()

        for item in raw:
            vid = item.get("vesselVisitId", "")
            line_key = self._coerce_line_key(item.get("lineKey"))
            if line_key not in target_set:
                continue
            matched_line_keys.add(line_key)

            eta = self._parse_dt(item.get("eta"))
            etd = self._parse_dt(item.get("etd"))

            vessel_info = item.get("vesselInfo") or {}
            vessel_id_str = vessel_info.get("id") or vid
            vessel_name = vessel_info.get("name") or ""

            vessels[vid] = Vessel(
                vessel_id=vessel_id_str,
                vessel_name=vessel_name,
                voyage_id=vid,
                eta=eta or datetime.now(),
                etd=etd or (datetime.now() + timedelta(hours=48)),
                berth_id="",
            )

        missing = target_set - matched_line_keys
        if missing:
            logger.warning(f"TOSLoader: VesselVisit 中未找到航线号: {sorted(missing)}")
        else:
            logger.info(
                f"TOSLoader: 加载船舶 {len(vessels)} 艘"
                f" (lineKeys: {sorted(target_set)})"
            )
        return vessels

    def load_discharge_containers(
        self,
        line_keys: List[int],
        vessels: Dict[str, Vessel],
    ) -> List[Container]:
        """
        从 BoundList JSON 的 Inbound 列表中提取目标航线的卸船箱（进口箱）。

        过滤条件:
          serviceLineKey in line_keys  AND  boundType == 1  AND  visitType == 1

        Parameters
        ----------
        line_keys : 目标航线号列表（对应 container.serviceLineKey，int）
        vessels   : load_vessels() 的返回值，用于补充 eta/etd/vessel_id

        Returns
        -------
        List[Container]  (business_type 固定为 IMPORT)
        """
        with open(self.bound_list_path, "r", encoding="utf-8") as f:
            raw: dict = json.load(f)

        inbound: List[dict] = raw.get("Inbound", [])
        target_set = {
            lk for lk in (self._coerce_line_key(value) for value in line_keys)
            if lk is not None
        }
        containers: List[Container] = []

        for item in inbound:
            c_raw = item.get("container")
            if not c_raw:
                continue

            service_line_key = self._coerce_line_key(c_raw.get("serviceLineKey"))
            if service_line_key not in target_set:
                continue

            dto = item.get("boundListDTO") or {}
            visit_id = dto.get("visitId") or ""
            if dto.get("boundType") != 1:
                continue
            if dto.get("visitType") != 1:
                continue

            vessel = vessels.get(visit_id)
            iso_type = c_raw.get("containerISO") or ""
            is_gauge = any(self._coerce_bool(c_raw.get(key)) for key in (
                "oog",
                "overLongBack",
                "overLongFront",
                "overWidthLeft",
                "overWidthRight",
            ))

            containers.append(Container(
                container_id=c_raw.get("containerId") or dto.get("contrId", ""),
                size=self._parse_size(c_raw.get("containerSize", 1)),
                container_type=self._parse_container_type(
                    iso_type,
                    c_raw.get("powerRequired", 0),
                ),
                weight_class=self._parse_weight_class(
                    c_raw.get("weight"),
                    c_raw.get("freightKind"),
                ),
                business_type=BusinessType.IMPORT,
                voyage_id=visit_id,
                line_key=service_line_key,
                vessel_id=vessel.vessel_id if vessel else visit_id,
                eta=vessel.eta if vessel else datetime.now(),
                etd=vessel.etd if vessel else None,
                iso_type=iso_type,
                category=c_raw.get("category"),
                pod=c_raw.get("pod"),
                cattier_kind=c_raw.get("cattierKind"),
                trade_code=c_raw.get("tradeCode"),
                freight_kind=c_raw.get("freightKind"),
                owner_company=c_raw.get("ownerCompany"),
                line_company=c_raw.get("lineCompany"),
                truck_company=c_raw.get("truckCompany"),
                belonger_company=c_raw.get("gradesCompany"),
                work_type=c_raw.get("workType"),
                bol=c_raw.get("bol"),
                damage_code=c_raw.get("damageType"),
                raw_weight=c_raw.get("weight"),
                is_reefer=self._coerce_bool(c_raw.get("powerRequired")) or "RF" in iso_type.upper(),
                is_hazardous=self._coerce_bool(c_raw.get("bHazardous")),
                is_damage=self._coerce_bool(c_raw.get("damage")) or bool(c_raw.get("damageType")),
                is_high=self._coerce_bool(c_raw.get("overHeight")),
                is_gauge=is_gauge,
                is_dirty=self._coerce_bool(c_raw.get("dirty")),
            ))

        logger.info(
            f"TOSLoader: 加载卸船箱 {len(containers)} 个"
            f" (lineKeys: {sorted(target_set)})"
        )
        return containers

    def load_external_allocation_groups(
        self,
        line_keys: List[int],
        vessels: Dict[str, Vessel],
    ) -> List[AllocationGroup]:
        """
        预留 type=2 的外部分配组读取入口。

        后续接口接入后，在这里把接口返回值转换成 AllocationGroup 列表。
        每个外部分配组至少需要提供:
          - group_id
          - business_type
          - size
          - container_type
          - weight_class
          - line_key
          - column_demand

        如果接口已经返回 filter，可放到:
          group.group_attributes["filter"]

        这样后续两阶段算法和最终 range_plan 格式都不用再改。
        """
        normalized_line_keys = [
            lk for lk in (self._coerce_line_key(value) for value in line_keys)
            if lk is not None
        ]
        raise NotImplementedError(
            "type=2 需要从外部分配组接口读取数据；接口未提供，"
            f"已预留 load_external_allocation_groups(line_keys={normalized_line_keys})"
        )

    def build_planning_horizon(
        self, vessels: Dict[str, Vessel]
    ) -> Tuple[datetime, datetime]:
        """
        从多艘船信息推断规划时间窗口。

        Returns
        -------
        (horizon_start, horizon_end)
          horizon_start = 所有船中最早的 ETA
          horizon_end   = 所有船中最晚的 ETD
        """
        all_v = list(vessels.values())
        if not all_v:
            now = datetime.now()
            return now, now + timedelta(hours=48)
        horizon_start = min(v.eta for v in all_v)
        horizon_end = max(v.etd for v in all_v)
        if horizon_end <= horizon_start:
            horizon_end = horizon_start + timedelta(hours=48)
        logger.info(
            f"TOSLoader: 规划时间窗口 {horizon_start} → {horizon_end}"
        )
        return horizon_start, horizon_end

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _parse_dt(dt_str: Optional[str]) -> Optional[datetime]:
        """解析 TOS 时间字符串，兼容含时区后缀的格式。"""
        if not dt_str:
            return None
        # 去掉时区部分 (+00:00 / Z)
        cleaned = dt_str.split("+")[0].split("Z")[0].strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
        logger.warning(f"TOSLoader: 无法解析时间字符串: {dt_str!r}")
        return None

    @staticmethod
    def _coerce_line_key(raw_value: Any) -> Optional[int]:
        """将 lineKey/serviceLineKey 统一解析为 int。"""
        if raw_value is None:
            return None
        if isinstance(raw_value, bool):
            return None
        if isinstance(raw_value, int):
            return raw_value
        if isinstance(raw_value, float):
            return int(raw_value) if raw_value.is_integer() else None

        text = str(raw_value).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @staticmethod
    def _coerce_bool(raw_value: Any) -> bool:
        if isinstance(raw_value, bool):
            return raw_value
        if raw_value is None:
            return False
        if isinstance(raw_value, (int, float)):
            return raw_value != 0
        return str(raw_value).strip().lower() in {"1", "true", "yes", "y"}

    @classmethod
    def _parse_size(cls, size_int: int) -> ContainerSize:
        return cls._SIZE_MAP.get(size_int, ContainerSize.SIZE_20)

    @classmethod
    def _parse_container_type(cls, iso_code: str, power_required: int) -> ContainerType:
        """
        containerISO 后缀优先; powerRequired=1 强制识别为冷藏箱。
        """
        if power_required:
            return ContainerType.REEFER
        code = (iso_code or "").upper()
        for suffix, ctype in cls._ISO_TYPE_MAP.items():
            if suffix in code:
                return ctype
        return ContainerType.DRY

    @classmethod
    def _parse_weight_class(
        cls,
        weight: Optional[float],
        freight_kind: Optional[int],
    ) -> WeightClass:
        """
        freightKind=3 → 空箱 (EMPTY)
        weight > HEAVY_WEIGHT_THRESHOLD → 重箱 (HEAVY)
        其余 → 轻箱 (LIGHT)
        """
        if freight_kind == 3:
            return WeightClass.EMPTY
        if weight is None or weight <= 0:
            return WeightClass.EMPTY
        if weight > cls.HEAVY_WEIGHT_THRESHOLD:
            return WeightClass.HEAVY
        return WeightClass.LIGHT


# ═══════════════════════════════════════════════════════════════════════════════
# Section M: Main Runner
# ═══════════════════════════════════════════════════════════════


def run_plan(
    line_keys: List[int],
    type: int = 1,
    apply_to_yard: bool = False,
) -> PlanningResult:
    """
    规划入口：传入一个或多个航线号，完成卸船箱堆场分配规划。

    Parameters
    ----------
    line_keys    : 目标航线号列表，如 [469144, 471924]
    type         : 1 → 读取航线箱子并自动划分分配组
                   2 → 从外部分配组接口读取分配组，直接进入两阶段算法
    apply_to_yard: True → 将规划结果写回 YardSpace，实时占用对应槽位

    Returns
    -------
    PlanningResult
      result.metrics["range_plan"] = {"data": [...]}，其中每项为分配组范围。
    """
    if type not in (1, 2):
        raise ValueError(f"type 只能为 1 或 2，当前为: {type!r}")

    loader = TOSLoader()
    normalized_line_keys: List[int] = []
    for raw_key in line_keys:
        line_key = loader._coerce_line_key(raw_key)
        if line_key is None:
            raise ValueError(f"无效的 lineKey: {raw_key!r}")
        normalized_line_keys.append(line_key)

    print("=" * 70)
    print(f"  堆场规划  type={type}  lineKeys: {normalized_line_keys}")
    print("=" * 70)

    # ── 步骤 1: 从 TOS JSON 加载船舶信息 ────────────────────────────────────
    vessels = loader.load_vessels(normalized_line_keys)
    horizon_start, horizon_end = loader.build_planning_horizon(vessels)

    containers: List[Container] = []
    external_groups: List[AllocationGroup] = []

    if type == 1:
        # ── 步骤 2A: 提取卸船箱，再由算法自动划分分配组 ─────────────────────
        containers = loader.load_discharge_containers(normalized_line_keys, vessels)
        if not containers:
            logger.warning("未加载到任何卸船箱，规划终止")
            return PlanningResult(
                run_id=f"PLAN-EMPTY",
                timestamp=datetime.now(),
                mode=PlannerMode.FULL_PLAN,
            )
    else:
        # ── 步骤 2B: 从外部分配组接口读取分配组（接口暂未接入） ─────────────
        external_groups = loader.load_external_allocation_groups(
            normalized_line_keys,
            vessels,
        )
        if not external_groups:
            logger.warning("未加载到任何外部分配组，规划终止")
            return PlanningResult(
                run_id=f"PLAN-EMPTY",
                timestamp=datetime.now(),
                mode=PlannerMode.FULL_PLAN,
            )

    # ── 步骤 3: 加载堆场实时状态 ─────────────────────────────────────────────
    from useable_space import YardSpace
    yard = YardSpace.load()
    yard.print_summary()

    # ── 步骤 4: 配置箱区进/出口属性（奇数编号=进口，偶数=出口） ─────────────
    block_ids = sorted({k[0] for k in yard.stacks.keys()})

    def _serial(block_id: str) -> int:
        digits = "".join(c for c in block_id if c.isdigit())
        return int(digits) if digits else 0

    block_business_types = {
        bid: (BusinessType.IMPORT if _serial(bid) % 2 == 1 else BusinessType.EXPORT)
        for bid in block_ids
    }

    # ── 步骤 5: 执行两阶段规划 ──────────────────────────────────────────────
    planner = YardPlanner()
    if type == 1:
        result = planner.plan_with_yard_space(
            yard=yard,
            containers=containers,
            block_business_types=block_business_types,
            mode=PlannerMode.FULL_PLAN,
            apply_to_yard=apply_to_yard,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
        )
    else:
        result = planner.plan_groups_with_yard_space(
            yard=yard,
            groups=external_groups,
            block_business_types=block_business_types,
            mode=PlannerMode.FULL_PLAN,
            apply_to_yard=apply_to_yard,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
        )

    # ── 步骤 6: 打印分配组范围结果 ─────────────────────────────────────────
    range_items = result.metrics.get("range_plan", {}).get("data", [])
    if range_items:
        print(f"\n  分配组范围结果 ({len(range_items)} 个分配组):")
        for item in range_items[:20]:
            print(
                f"    groupId={item['groupId']} groupKey={item['groupKey']} "
                f"ranges={item['rangeList']} filter={item.get('filter', {})}"
            )
        if len(range_items) > 20:
            print(f"    ... 共 {len(range_items)} 个分配组")
    else:
        print("\n  无分配组范围结果")

    return result


if __name__ == "__main__":
    # 传入目标航线号（int，可多个），按需修改
    run_plan(line_keys=[469144], type=1)