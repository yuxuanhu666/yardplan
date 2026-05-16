from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("YardPlanner")


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

    GROUP_ONLY = "group_only"
    ALLOCATE_ONLY = "allocate_only"
    FULL_PLAN = "full_plan"


class FlowType(Enum):
    """
    Four yard space flows (Section 3 of spec).
    DISCHARGE: vessel -> yard (import in)
    LOADING:   yard -> vessel (export out)
    GATE_IN:   truck -> yard (export in)
    GATE_OUT:  yard -> truck (import out)
    """

    DISCHARGE = "discharge"
    LOADING = "loading"
    GATE_IN = "gate_in"
    GATE_OUT = "gate_out"


MAX_TIERS_PER_COLUMN = 6
MAX_SPLITS_PER_GROUP = 2
MIN_COLUMNS_PER_SPLIT = 1


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

    voyage_id: str
    line_key: Optional[int]
    vessel_id: str

    eta: datetime
    etd: Optional[datetime] = None

    earliest_pickup: Optional[datetime] = None
    latest_pickup: Optional[datetime] = None

    destination_port: Optional[str] = None
    receiving_start: Optional[datetime] = None
    receiving_end: Optional[datetime] = None

    consignee: Optional[str] = None
    pickup_party: Optional[str] = None

    current_block: Optional[str] = None
    current_bay: Optional[int] = None
    current_row: Optional[str] = None
    current_tier: Optional[int] = None

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
        return self.size in (ContainerSize.SIZE_40, ContainerSize.SIZE_45)

    def is_edge_only(self) -> bool:
        return self.size == ContainerSize.SIZE_45


@dataclass
class Bay:
    """
    A single bay within a yard block/area.
    In this planner, "column" refers to a (bay, row) slot stack.
    """

    bay_id: str
    bay_number: int
    yard_area_id: str
    total_columns: int
    occupied_columns: int = 0
    size_lock: Optional[ContainerSize] = None
    is_in_large_bay: bool = False

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
    """

    pair_id: str
    yard_area_id: str
    bay_a: Bay
    bay_b: Bay
    is_edge_pair: bool = False

    @property
    def total_columns(self) -> int:
        return min(self.bay_a.total_columns, self.bay_b.total_columns)

    @property
    def occupied_columns(self) -> int:
        return max(self.bay_a.occupied_columns, self.bay_b.occupied_columns)

    @property
    def free_columns(self) -> int:
        return max(0, self.total_columns - self.occupied_columns)

    def can_accept_45ft(self) -> bool:
        return self.is_edge_pair and self.free_columns > 0

    def can_accept_40ft(self) -> bool:
        return self.free_columns > 0


@dataclass
class YardArea:
    """
    A logical yard area (block) dedicated to either import or export.
    Contains an ordered list of bays and pre-computed large bay pairs.
    """

    area_id: str
    business_type: BusinessType
    bays: List[Bay] = field(default_factory=list)
    large_bay_pairs: List[LargeBayPair] = field(default_factory=list)
    supported_sizes: Set[ContainerSize] = field(
        default_factory=lambda: {
            ContainerSize.SIZE_20,
            ContainerSize.SIZE_40,
            ContainerSize.SIZE_45,
        }
    )
    max_stack_height: int = MAX_TIERS_PER_COLUMN
    distance_to_gate: float = 0.0
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
        return sum(
            p.free_columns for p in self.large_bay_pairs if p.can_accept_40ft()
        )

    def get_edge_pairs(self) -> List[LargeBayPair]:
        return [p for p in self.large_bay_pairs if p.is_edge_pair]

    def get_bay_by_number(self, bay_number: int) -> Optional[Bay]:
        for bay in self.bays:
            if bay.bay_number == bay_number:
                return bay
        return None


@dataclass
class Vessel:
    vessel_id: str
    vessel_name: str
    voyage_id: str
    eta: datetime
    etd: datetime
    berth_id: str
    discharge_containers: List[Container] = field(default_factory=list)
    loading_containers: List[Container] = field(default_factory=list)


@dataclass
class YardFlow:
    flow_id: str
    flow_type: FlowType
    business_type: BusinessType
    yard_area_id: Optional[str]
    container_count: int
    column_delta: int
    expected_time: datetime
    voyage_id: Optional[str] = None
    certainty: float = 1.0


@dataclass
class AllocationGroup:
    """
    Stage1/Stage2 共用的分配组。

    - `column_demand` 表示堆场容量占用，仍用于容量约束与 Stage2。
    - `container_count` 表示箱数，Stage1 场桥工作量（moves / 箱次）必须基于它计算，
      不能再用 `column_demand` 近似替代。
    """

    group_id: str
    business_type: BusinessType
    size: ContainerSize
    container_type: ContainerType
    weight_class: WeightClass
    voyage_id: str
    line_key: Optional[int]
    group_attributes: Dict[str, Any] = field(default_factory=dict)
    containers: List[Container] = field(default_factory=list)
    column_demand: int = 0
    container_count: int = 0
    earliest_arrival: Optional[datetime] = None
    latest_departure: Optional[datetime] = None
    is_split: bool = False
    parent_group_id: Optional[str] = None
    split_index: int = 0

    def __post_init__(self) -> None:
        if self.containers:
            self.container_count = len(self.containers)
            return
        self.container_count = max(0, int(self.container_count or 0))

    @property
    def is_large_container_group(self) -> bool:
        return self.size in (ContainerSize.SIZE_40, ContainerSize.SIZE_45)

    @property
    def is_edge_only(self) -> bool:
        return self.size == ContainerSize.SIZE_45


@dataclass
class AreaAssignment:
    assignment_id: str
    group_id: str
    yard_area_id: str
    column_demand: int
    split_index: int = 0
    is_partial: bool = False


@dataclass
class BayColumnAllocation:
    allocation_id: str
    group_id: str
    yard_area_id: str
    business_type: BusinessType
    size: ContainerSize
    split_index: int = 0
    bay_column_details: List[Tuple[Any, int]] = field(default_factory=list)
    bay_stack_details: List[Tuple[Any, int, int]] = field(default_factory=list)
    is_edge_placement: bool = False
    is_spanning: bool = False
    notes: str = ""


@dataclass
class PlanningResult:
    run_id: str
    timestamp: datetime
    mode: PlannerMode
    allocation_groups: List[AllocationGroup] = field(default_factory=list)
    area_assignments: List[AreaAssignment] = field(default_factory=list)
    bay_column_allocations: List[BayColumnAllocation] = field(default_factory=list)
    unassigned_groups: List[AllocationGroup] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "_DATA_DIR",
    "logger",
    "MAX_TIERS_PER_COLUMN",
    "MAX_SPLITS_PER_GROUP",
    "MIN_COLUMNS_PER_SPLIT",
    "BusinessType",
    "ContainerSize",
    "ContainerType",
    "WeightClass",
    "PlannerMode",
    "FlowType",
    "Container",
    "Bay",
    "LargeBayPair",
    "YardArea",
    "Vessel",
    "YardFlow",
    "AllocationGroup",
    "AreaAssignment",
    "BayColumnAllocation",
    "PlanningResult",
]
