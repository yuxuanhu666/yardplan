from __future__ import annotations

import uuid
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from yardplan_core.models import (
    AllocationGroup,
    BusinessType,
    Container,
    ContainerSize,
    ContainerType,
    MAX_SPLITS_PER_GROUP,
    MAX_TIERS_PER_COLUMN,
    WeightClass,
    logger,
)


class ColumnDemandConverter:
    """
    Converts container counts to column demand.

    [REPLACEABLE MODULE] This module encapsulates all column-demand logic.
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
        effective_tiers = self._effective_tiers(weight_class, container_type)
        return max(1, -(-container_count // effective_tiers))

    def _effective_tiers(
        self,
        weight_class: WeightClass,
        container_type: ContainerType,
    ) -> int:
        if container_type == ContainerType.REEFER:
            return 3
        if weight_class == WeightClass.HEAVY:
            return 4
        if container_type in (ContainerType.OPEN_TOP, ContainerType.FLAT_RACK):
            return 2
        return self.max_tiers

    def convert_group(self, group: AllocationGroup) -> int:
        demand = self.compute_column_demand(
            container_count=group.container_count,
            size=group.size,
            weight_class=group.weight_class,
            container_type=group.container_type,
        )
        group.column_demand = demand
        return demand


GroupKeyFunc = Callable[[Container], Tuple]

DEFAULT_EXPORT_GROUP_KEYS: List[str] = [
    "line_key",
    "size",
    "container_type",
    "weight_class",
    "pod",
]

DEFAULT_IMPORT_GROUP_KEYS: List[str] = [
    "line_key",
    "size",
    "container_type",
    "weight_class",
    "consignee",
]


class GroupingConfig:
    def __init__(
        self,
        export_keys: Optional[List[str]] = None,
        import_keys: Optional[List[str]] = None,
        custom_key_funcs: Optional[Dict[str, Callable[[Container], Any]]] = None,
        log_group_details: bool = True,
    ):
        self.export_keys = export_keys or DEFAULT_EXPORT_GROUP_KEYS
        self.import_keys = import_keys or DEFAULT_IMPORT_GROUP_KEYS
        self.custom_key_funcs = custom_key_funcs or {}
        self.log_group_details = log_group_details

    def get_key(self, container: Container, key_name: str) -> Any:
        if key_name in self.custom_key_funcs:
            return self.custom_key_funcs[key_name](container)
        value = getattr(container, key_name, None)
        if isinstance(value, Enum):
            return value.value
        return value

    def make_group_key(
        self,
        container: Container,
        business_type: BusinessType,
    ) -> Tuple:
        keys = (
            self.export_keys
            if business_type == BusinessType.EXPORT
            else self.import_keys
        )
        return tuple(self.get_key(container, key) for key in keys)


class GroupingEngine:
    """
    Function 1: Groups containers into AllocationGroups.
    """

    def __init__(
        self,
        config: Optional[GroupingConfig] = None,
        demand_converter: Optional[ColumnDemandConverter] = None,
    ):
        self.config = config or GroupingConfig()
        self.demand_converter = demand_converter or ColumnDemandConverter()

    def group_containers(
        self,
        containers: List[Container],
    ) -> List[AllocationGroup]:
        import_containers = [
            container
            for container in containers
            if container.business_type == BusinessType.IMPORT
        ]
        export_containers = [
            container
            for container in containers
            if container.business_type == BusinessType.EXPORT
        ]

        groups: List[AllocationGroup] = []
        groups.extend(self._group_by_type(import_containers, BusinessType.IMPORT))
        groups.extend(self._group_by_type(export_containers, BusinessType.EXPORT))

        logger.info(
            f"Grouping complete: {len(groups)} groups "
            f"({sum(1 for g in groups if g.business_type == BusinessType.IMPORT)} import, "
            f"{sum(1 for g in groups if g.business_type == BusinessType.EXPORT)} export)"
        )
        if self.config.log_group_details:
            self._log_allocation_groups(groups)
        return groups

    def _log_allocation_groups(self, groups: List[AllocationGroup]) -> None:
        if not groups:
            return

        logger.info("Allocation groups after grouping (%s):", len(groups))
        for index, group in enumerate(
            sorted(groups, key=lambda item: (-item.container_count, item.group_id)),
            start=1,
        ):
            size_value = (
                group.size.value if hasattr(group.size, "value") else str(group.size)
            )
            business_value = (
                group.business_type.value
                if hasattr(group.business_type, "value")
                else str(group.business_type)
            )
            pod = self._group_pod_summary(group)
            attrs = ", ".join(
                f"{key}={value}"
                for key, value in sorted(group.group_attributes.items())
            )
            logger.info(
                "  [%02d] %s  %s  size=%s  containers=%s  columns=%s  pod=%s  %s",
                index,
                group.group_id,
                business_value,
                size_value,
                group.container_count,
                group.column_demand,
                pod,
                attrs or "(no group_attributes)",
            )

    @staticmethod
    def _group_pod_summary(group: AllocationGroup) -> str:
        pod_from_attrs = group.group_attributes.get("pod")
        if pod_from_attrs not in (None, ""):
            return str(pod_from_attrs)

        pods = sorted(
            {
                str(container.pod)
                for container in group.containers
                if container.pod not in (None, "")
            }
        )
        if not pods:
            return "-"
        if len(pods) == 1:
            return pods[0]
        return f"{pods[0]}(+{len(pods) - 1})"

    def _group_by_type(
        self,
        containers: List[Container],
        business_type: BusinessType,
    ) -> List[AllocationGroup]:
        if not containers:
            return []

        bucket: Dict[Tuple, List[Container]] = defaultdict(list)
        for container in containers:
            key = self.config.make_group_key(container, business_type)
            bucket[key].append(container)

        groups: List[AllocationGroup] = []
        for key, members in bucket.items():
            groups.append(self._build_group(key, members, business_type))
        return groups

    def _build_group(
        self,
        key: Tuple,
        members: List[Container],
        business_type: BusinessType,
    ) -> AllocationGroup:
        representative = members[0]
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

        arrivals = [
            container.eta
            if business_type == BusinessType.IMPORT
            else (container.receiving_start or container.eta)
            for container in members
            if (
                container.eta
                if business_type == BusinessType.IMPORT
                else (container.receiving_start or container.eta)
            )
        ]
        departures = [
            container.latest_pickup
            if business_type == BusinessType.IMPORT and container.latest_pickup
            else (container.etd if container.etd else None)
            for container in members
        ]
        departures = [departure for departure in departures if departure is not None]

        if arrivals:
            group.earliest_arrival = min(arrivals)
        if departures:
            group.latest_departure = max(departures)

        self.demand_converter.convert_group(group)
        return group

    def split_group(
        self,
        group: AllocationGroup,
        split_column_demands: List[int],
    ) -> List[AllocationGroup]:
        assert len(split_column_demands) <= MAX_SPLITS_PER_GROUP + 1, (
            f"Cannot split into more than {MAX_SPLITS_PER_GROUP + 1} parts"
        )
        assert sum(split_column_demands) == group.column_demand, (
            "Split demands must sum to original demand"
        )

        sub_groups: List[AllocationGroup] = []
        containers = list(group.containers)
        total = max(group.container_count, len(containers))
        total_demand = group.column_demand
        remaining = total

        for idx, demand in enumerate(split_column_demands):
            if idx < len(split_column_demands) - 1:
                count = max(1, round(total * demand / total_demand))
                count = min(count, remaining)
            else:
                count = remaining

            assigned_count = min(count, len(containers))
            sub_containers = containers[:assigned_count]
            containers = containers[assigned_count:]
            remaining = max(0, remaining - count)

            sub_groups.append(
                AllocationGroup(
                    group_id=f"{group.group_id}-S{idx + 1}",
                    business_type=group.business_type,
                    size=group.size,
                    container_type=group.container_type,
                    weight_class=group.weight_class,
                    voyage_id=group.voyage_id,
                    line_key=group.line_key,
                    group_attributes=group.group_attributes.copy(),
                    containers=sub_containers,
                    container_count=count,
                    column_demand=demand,
                    earliest_arrival=group.earliest_arrival,
                    latest_departure=group.latest_departure,
                    is_split=True,
                    parent_group_id=group.group_id,
                    split_index=idx + 1,
                )
            )

        return sub_groups


__all__ = [
    "ColumnDemandConverter",
    "GroupKeyFunc",
    "DEFAULT_EXPORT_GROUP_KEYS",
    "DEFAULT_IMPORT_GROUP_KEYS",
    "GroupingConfig",
    "GroupingEngine",
]
