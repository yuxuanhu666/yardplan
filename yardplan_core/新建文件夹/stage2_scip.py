from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from yardplan_core.models import (
    AllocationGroup,
    AreaAssignment,
    BayColumnAllocation,
    ContainerSize,
    YardArea,
    logger,
)
from yardplan_core.simultaneous import (
    groups_have_simultaneous_loading_conflict,
    simultaneous_loading_safety_gap_bays,
)

Atom = Tuple[int, int, int]
BaySpec = Any


def merge_same_parent_assignments_for_stage2(
    area_assignments: List[AreaAssignment],
    groups: Dict[str, AllocationGroup],
) -> Tuple[List[AreaAssignment], Set[Tuple[str, str]]]:
    """
    When Stage1 splits one logical parent into several AllocationGroups that land in the
    same yard area, Stage2 otherwise solves them as independent items and fragments bays.

    Merge those sibling assignments into one AreaAssignment keyed by `parent_group_id`
    (using the parent's AllocationGroup from `groups`) so the MILP optimizes a single
    demand vector and bay-axis region penalties apply to one block.

    Returns the merged assignment list and the set of (yard_area_id, parent_group_id)
    pairs that were merged (for post-run contiguity logging).
    """
    clusters: Dict[Tuple[str, str], List[AreaAssignment]] = defaultdict(list)
    for assignment in area_assignments:
        group = groups.get(assignment.group_id)
        if group is None or not group.parent_group_id:
            continue
        clusters[(assignment.yard_area_id, group.parent_group_id)].append(assignment)

    multi_keys = {key for key, bucket in clusters.items() if len(bucket) >= 2}
    merged_parent_area_keys: Set[Tuple[str, str]] = set()
    merged_assignments: List[AreaAssignment] = []
    processed_multi: Set[Tuple[str, str]] = set()

    for assignment in area_assignments:
        group = groups.get(assignment.group_id)
        if group is None:
            merged_assignments.append(assignment)
            continue

        parent_id = group.parent_group_id
        key: Optional[Tuple[str, str]] = (
            (assignment.yard_area_id, parent_id) if parent_id else None
        )

        if key and key in multi_keys:
            if key in processed_multi:
                continue

            bucket = clusters[key]
            parent = groups.get(parent_id) if parent_id else None
            sorted_bucket = sorted(bucket, key=lambda a: (a.split_index, a.group_id))

            if parent is None:
                logger.warning(
                    "Stage2 merge skipped: parent group %s not found (%s split "
                    "assignments in area %s)",
                    parent_id,
                    len(sorted_bucket),
                    assignment.yard_area_id,
                )
                processed_multi.add(key)
                merged_assignments.extend(sorted_bucket)
                continue

            ref_size = parent.size
            if any(groups[a.group_id].size != ref_size for a in sorted_bucket):
                logger.warning(
                    "Stage2 merge skipped: sibling size mismatch under parent %s in area %s",
                    parent_id,
                    assignment.yard_area_id,
                )
                processed_multi.add(key)
                merged_assignments.extend(sorted_bucket)
                continue

            processed_multi.add(key)
            merged_parent_area_keys.add(key)
            total_demand = sum(max(0, int(a.column_demand)) for a in sorted_bucket)
            merged_assignments.append(
                AreaAssignment(
                    assignment_id=f"ASN-{uuid.uuid4().hex[:8].upper()}",
                    group_id=parent_id,
                    yard_area_id=assignment.yard_area_id,
                    column_demand=total_demand,
                    split_index=0,
                    is_partial=any(a.is_partial for a in sorted_bucket),
                )
            )
            logger.info(
                "Stage2 merged %s sibling assignments into parent %s in area %s "
                "(column_demand=%s)",
                len(sorted_bucket),
                parent_id,
                assignment.yard_area_id,
                total_demand,
            )
            continue

        merged_assignments.append(assignment)

    return merged_assignments, merged_parent_area_keys


def _bays_from_bay_spec(bay_spec: BaySpec) -> List[int]:
    if isinstance(bay_spec, tuple):
        return [int(value) for value in bay_spec]
    return [int(bay_spec)]


def _unique_sorted_bays_from_allocation(alloc: BayColumnAllocation) -> List[int]:
    bays: Set[int] = set()
    for bay_spec, _count in alloc.bay_column_details:
        bays.update(_bays_from_bay_spec(bay_spec))
    for bay_spec, _start_stack, _end_stack in alloc.bay_stack_details:
        bays.update(_bays_from_bay_spec(bay_spec))
    return sorted(bays)


def _bay_axis_contiguity_stats(bays: Sequence[int]) -> Dict[str, Any]:
    if not bays:
        return {"bay_count": 0, "runs": 0, "hull_span": 0, "hull_slack": 0}
    sorted_bays = sorted(bays)
    runs = 1
    for index in range(1, len(sorted_bays)):
        if sorted_bays[index] != sorted_bays[index - 1] + 1:
            runs += 1
    hull_span = sorted_bays[-1] - sorted_bays[0] + 1
    hull_slack = hull_span - len(sorted_bays)
    return {
        "bay_count": len(sorted_bays),
        "runs": runs,
        "hull_span": hull_span,
        "hull_slack": hull_slack,
    }


def log_stage2_merged_parent_contiguity(
    allocations: List[BayColumnAllocation],
    merged_parent_area_keys: Set[Tuple[str, str]],
) -> None:
    """Log bay-axis fragmentation metrics for allocations produced from merged siblings."""
    if not merged_parent_area_keys:
        return

    for alloc in allocations:
        key = (alloc.yard_area_id, alloc.group_id)
        if key not in merged_parent_area_keys:
            continue

        bays = _unique_sorted_bays_from_allocation(alloc)
        stats = _bay_axis_contiguity_stats(bays)
        logger.info(
            "Stage2 merged-parent contiguity parent=%s area=%s bay_count=%s "
            "contiguous_runs=%s hull_span=%s hull_slack=%s (runs=1 and slack=0 is ideal)",
            alloc.group_id,
            alloc.yard_area_id,
            stats["bay_count"],
            stats["runs"],
            stats["hull_span"],
            stats["hull_slack"],
        )


@dataclass(frozen=True)
class Stage2PlacementOption:
    """One feasible tier placement candidate inside a yard area."""

    option_id: str
    bay_spec: BaySpec
    atoms: Tuple[Atom, ...]
    stack_index: int
    tier_index: int
    zone: str
    rank: int
    segment_capacity: int = 1
    segment_scarcity_cost: float = 0.0
    column_occupied_tiers: int = 0
    segment_occupied_tiers: int = 0


@dataclass
class Stage2ScipConfig:
    time_limit_seconds: float = 120.0
    unmet_weight: float = 1_000_000.0
    segment_weight: float = 1_000.0
    region_component_weight: float = 1_500.0
    # Each planned space should be compact: stacks must be contiguous within one
    # bay, and a continuing bay run must fill the previous bay before opening the
    # adjacent one.  We still allow multiple bay runs for one group when the yard
    # has unavoidable gaps; each run then has at most one underfilled tail bay.
    enforce_stack_contiguity: bool = True
    segment_min_fill_ratio: float = 0.65
    max_underfilled_segments_per_item: int = 2
    # Crane balance should only be a weak tie-breaker; it must not tear apart
    # a physically feasible contiguous parent-group block.
    crane_balance_weight: float = 1.0
    stability_weight: float = 0.001
    # Break stack-symmetry inside the same bay/pair: when two shapes are equally
    # compact, prefer lower stack indices so an empty bay is filled from column 1.
    stack_position_weight: float = 5.0
    # Prefer bays/pairs with more remaining empty columns. A nearly full bay is
    # still usable when necessary, but it should lose to a wider empty bay.
    segment_scarcity_weight: float = 250.0
    enforce_bay_run_contiguity: bool = False
    enforce_sequential_segment_fill: bool = True
    sequential_fill_max_tier_demand: int = 80
    enforce_segment_min_fill: bool = False
    stack_run_weight: float = 600.0
    isolated_cell_weight: float = 0.0
    underfilled_segment_weight: float = 0.0
    soft_geometric_region_axis: bool = True
    simultaneous_loading_min_bay_gap: int = 4


class ScipStage2BayAllocator:
    """
    Greedy implementation of stage 2 bay/column allocation.

    Allocation is built independently per yard area. Every decision consumes
    bottom-level atoms `(bay_number, stack_index)`, so 20/40/45ft placements
    share the same physical resources and cannot overlap.
    """

    def __init__(self, config: Optional[Stage2ScipConfig] = None):
        self.config = config or Stage2ScipConfig()
        self.placement_progress: List[Dict[str, Any]] = []
        self._progress_global_demand = 0
        self._progress_global_placed = 0

    def allocate(
        self,
        area_assignments: List[AreaAssignment],
        groups: Dict[str, AllocationGroup],
        yard_areas: Dict[str, YardArea],
    ) -> List[BayColumnAllocation]:
        self.placement_progress = []
        self._progress_global_demand = 0
        self._progress_global_placed = 0
        by_area: Dict[str, List[Tuple[AreaAssignment, AllocationGroup]]] = defaultdict(list)
        for assignment in area_assignments:
            group = groups.get(assignment.group_id)
            if group is None or assignment.yard_area_id not in yard_areas:
                continue
            by_area[assignment.yard_area_id].append((assignment, group))

        self._progress_global_demand = sum(
            max(0, self._tier_demand(assignment, group))
            for items in by_area.values()
            for assignment, group in items
        )

        allocations: List[BayColumnAllocation] = []
        for area_id in sorted(by_area):
            area = yard_areas[area_id]
            allocations.extend(self._allocate_area(area, by_area[area_id]))
        return allocations

    def _allocate_area(
        self,
        area: YardArea,
        items: List[Tuple[AreaAssignment, AllocationGroup]],
    ) -> List[BayColumnAllocation]:
        return self._allocate_area_greedy(area, items)

    def _allocate_area_greedy(
        self,
        area: YardArea,
        items: List[Tuple[AreaAssignment, AllocationGroup]],
    ) -> List[BayColumnAllocation]:
        if not items:
            return []

        items = sorted(
            items,
            key=lambda item: (
                item[0].yard_area_id,
                item[0].split_index,
                item[0].group_id,
                item[0].assignment_id,
            ),
        )
        item_keys = [
            self._item_key(assignment, index)
            for index, (assignment, _group) in enumerate(items)
        ]
        options_by_size = self._build_options(area)
        column_base_tiers = self._column_base_tiers(options_by_size)

        selected_by_item: Dict[str, List[Stage2PlacementOption]] = {
            item_key: []
            for item_key in item_keys
        }
        occupied_atoms: Set[Atom] = set()
        used_20ft_bays: Set[int] = set(
            int(value)
            for value in getattr(area, "_stage2_existing_20ft_bays", set())
        )
        used_large_bays: Set[int] = set(
            int(value)
            for value in getattr(area, "_stage2_existing_large_bays", set())
        )
        column_groups: Dict[Tuple[BaySpec, int], List[AllocationGroup]] = defaultdict(list)
        column_sizes: Dict[Tuple[BaySpec, int], ContainerSize] = {}
        column_selected_by_tier: Dict[
            Tuple[BaySpec, int],
            Dict[int, AllocationGroup],
        ] = defaultdict(dict)
        selected_options_by_group: List[Tuple[AllocationGroup, Stage2PlacementOption]] = []

        item_contexts = [
            (
                item_key,
                assignment,
                group,
                self._tier_demand(assignment, group),
            )
            for item_key, (assignment, group) in zip(item_keys, items)
        ]
        area_total_demand = sum(
            max(0, demand)
            for _item_key, _assignment, _group, demand in item_contexts
        )
        area_placed = 0
        placement_order = sorted(
            item_contexts,
            key=lambda item: self._greedy_item_key(item[1], item[2], item[3]),
        )

        for item_key, assignment, group, demand in placement_order:
            if demand <= 0:
                continue

            selected = selected_by_item[item_key]
            selected_set: Set[Stage2PlacementOption] = set(selected)
            candidate_options = options_by_size.get(group.size, [])

            while len(selected) < demand:
                choice: Optional[Stage2PlacementOption] = None
                for option in sorted(
                    candidate_options,
                    key=lambda candidate: self._greedy_option_key(
                        candidate,
                        selected,
                    ),
                ):
                    if option in selected_set:
                        continue
                    if not self._greedy_option_feasible(
                        option=option,
                        group=group,
                        occupied_atoms=occupied_atoms,
                        used_20ft_bays=used_20ft_bays,
                        used_large_bays=used_large_bays,
                        column_groups=column_groups,
                        column_sizes=column_sizes,
                        column_selected_by_tier=column_selected_by_tier,
                        column_base_tiers=column_base_tiers,
                        selected_options_by_group=selected_options_by_group,
                    ):
                        continue
                    choice = option
                    break

                if choice is None:
                    break

                selected.append(choice)
                selected_set.add(choice)
                self._record_greedy_option(
                    option=choice,
                    group=group,
                    occupied_atoms=occupied_atoms,
                    used_20ft_bays=used_20ft_bays,
                    used_large_bays=used_large_bays,
                    column_groups=column_groups,
                    column_sizes=column_sizes,
                    column_selected_by_tier=column_selected_by_tier,
                )
                selected_options_by_group.append((group, choice))
                area_placed += 1
                self._progress_global_placed += 1
                self.placement_progress.append(
                    {
                        "step": len(self.placement_progress) + 1,
                        "areaId": area.area_id,
                        "groupId": group.group_id,
                        "businessType": getattr(group.business_type, "value", group.business_type),
                        "size": getattr(group.size, "value", group.size),
                        "baySpec": choice.bay_spec,
                        "stackIndex": choice.stack_index,
                        "tier": choice.tier_index,
                        "areaPlaced": area_placed,
                        "areaDemand": area_total_demand,
                        "areaRemaining": max(0, area_total_demand - area_placed),
                        "globalPlaced": self._progress_global_placed,
                        "globalDemand": self._progress_global_demand,
                        "globalRemaining": max(
                            0,
                            self._progress_global_demand - self._progress_global_placed,
                        ),
                    }
                )

            if len(selected) < demand:
                logger.warning(
                    "Stage2 greedy area=%s group=%s placed=%s/%s unmet=%s",
                    area.area_id,
                    group.group_id,
                    len(selected),
                    demand,
                    demand - len(selected),
                )

        allocations: List[BayColumnAllocation] = []
        status = "greedy"
        for item_key, assignment, group, demand in item_contexts:
            selected = selected_by_item[item_key]
            candidate_options = options_by_size.get(group.size, [])
            unmet_count = max(0, demand - len(selected))
            if unmet_count > 0:
                self._log_unmet_diagnostics(
                    area=area,
                    assignment=assignment,
                    group=group,
                    demand=demand,
                    unmet_count=unmet_count,
                    candidate_options=candidate_options,
                    selected=selected,
                    status=status,
                )
            allocations.append(
                self._build_allocation(
                    area=area,
                    assignment=assignment,
                    group=group,
                    selected=selected,
                    demand=demand,
                    unmet_count=unmet_count,
                    status=status,
                )
            )
        return allocations

    def _record_greedy_option(
        self,
        *,
        option: Stage2PlacementOption,
        group: AllocationGroup,
        occupied_atoms: Set[Atom],
        used_20ft_bays: Set[int],
        used_large_bays: Set[int],
        column_groups: Dict[Tuple[BaySpec, int], List[AllocationGroup]],
        column_sizes: Dict[Tuple[BaySpec, int], ContainerSize],
        column_selected_by_tier: Dict[Tuple[BaySpec, int], Dict[int, AllocationGroup]],
    ) -> None:
        occupied_atoms.update(option.atoms)
        bays = _bays_from_bay_spec(option.bay_spec)
        if group.size == ContainerSize.SIZE_20:
            used_20ft_bays.update(bays)
        else:
            used_large_bays.update(bays)

        column_key = (option.bay_spec, option.stack_index)
        if group not in column_groups[column_key]:
            column_groups[column_key].append(group)
        column_sizes[column_key] = group.size
        column_selected_by_tier[column_key][option.tier_index] = group

    @staticmethod
    def _column_base_tiers(
        options_by_size: Dict[ContainerSize, List[Stage2PlacementOption]],
    ) -> Dict[Tuple[BaySpec, int], int]:
        base_tiers: Dict[Tuple[BaySpec, int], int] = {}
        for options in options_by_size.values():
            for option in options:
                column_key = (option.bay_spec, option.stack_index)
                current = base_tiers.get(column_key)
                if current is None or option.tier_index < current:
                    base_tiers[column_key] = option.tier_index
        return base_tiers

    def _greedy_item_key(
        self,
        assignment: AreaAssignment,
        group: AllocationGroup,
        demand: int,
    ) -> Tuple[int, int, int, str, int, str, str]:
        size_priority = {
            ContainerSize.SIZE_45: 0,
            ContainerSize.SIZE_40: 1,
            ContainerSize.SIZE_20: 2,
        }.get(group.size, 3)
        return (
            size_priority,
            self._weight_rank(group),
            -int(demand),
            assignment.yard_area_id,
            assignment.split_index,
            group.group_id,
            assignment.assignment_id,
        )

    def _log_unmet_diagnostics(
        self,
        *,
        area: YardArea,
        assignment: AreaAssignment,
        group: AllocationGroup,
        demand: int,
        unmet_count: int,
        candidate_options: List[Stage2PlacementOption],
        selected: List[Stage2PlacementOption],
        status: str,
    ) -> None:
        candidate_columns = {
            (option.bay_spec, option.stack_index)
            for option in candidate_options
        }
        candidate_segments = {option.bay_spec for option in candidate_options}
        candidate_bays = {
            bay
            for option in candidate_options
            for bay in _bays_from_bay_spec(option.bay_spec)
        }
        selected_columns = {
            (option.bay_spec, option.stack_index)
            for option in selected
        }
        selected_segments = {option.bay_spec for option in selected}
        tier_counts_by_segment: Dict[BaySpec, int] = defaultdict(int)
        column_counts_by_segment: Dict[BaySpec, Set[int]] = defaultdict(set)
        for option in candidate_options:
            tier_counts_by_segment[option.bay_spec] += 1
            column_counts_by_segment[option.bay_spec].add(option.stack_index)

        densest_segments = sorted(
            (
                (
                    self._segment_name(segment),
                    len(column_counts_by_segment[segment]),
                    tier_count,
                )
                for segment, tier_count in tier_counts_by_segment.items()
            ),
            key=lambda item: (-item[2], item[0]),
        )[:8]
        logger.warning(
            "Stage2 unmet diagnostic area=%s group=%s status=%s "
            "tier_demand=%s unmet=%s container_count=%s assignment_columns=%s "
            "group_columns=%s size=%s weight=%s candidate_tiers=%s "
            "candidate_columns=%s candidate_segments=%s candidate_bays=%s "
            "selected_tiers=%s selected_columns=%s selected_segments=%s "
            "top_candidate_segments=%s",
            area.area_id,
            group.group_id,
            status,
            demand,
            unmet_count,
            group.container_count,
            assignment.column_demand,
            group.column_demand,
            getattr(group.size, "value", group.size),
            getattr(group.weight_class, "value", group.weight_class),
            len(candidate_options),
            len(candidate_columns),
            len(candidate_segments),
            len(candidate_bays),
            len(selected),
            len(selected_columns),
            len(selected_segments),
            densest_segments,
        )

    def _greedy_option_feasible(
        self,
        *,
        option: Stage2PlacementOption,
        group: AllocationGroup,
        occupied_atoms: Set[Atom],
        used_20ft_bays: Set[int],
        used_large_bays: Set[int],
        column_groups: Dict[Tuple[BaySpec, int], List[AllocationGroup]],
        column_sizes: Dict[Tuple[BaySpec, int], ContainerSize],
        column_selected_by_tier: Dict[Tuple[BaySpec, int], Dict[int, AllocationGroup]],
        column_base_tiers: Optional[Dict[Tuple[BaySpec, int], int]] = None,
        selected_options_by_group: Optional[
            List[Tuple[AllocationGroup, Stage2PlacementOption]]
        ] = None,
    ) -> bool:
        if any(atom in occupied_atoms for atom in option.atoms):
            return False

        bays = set(_bays_from_bay_spec(option.bay_spec))
        if group.size == ContainerSize.SIZE_20:
            if bays & used_large_bays:
                return False
        elif bays & used_20ft_bays:
            return False

        if not self._simultaneous_loading_gap_feasible(
            option=option,
            group=group,
            selected_options_by_group=selected_options_by_group or [],
        ):
            return False

        column_key = (option.bay_spec, option.stack_index)
        existing_size = column_sizes.get(column_key)
        if existing_size is not None and existing_size != group.size:
            return False

        for existing_group in column_groups.get(column_key, []):
            if not self._groups_column_compatible(group, existing_group):
                return False

        tiers = column_selected_by_tier.get(column_key, {})
        base_tier = (
            column_base_tiers.get(column_key, option.tier_index)
            if column_base_tiers is not None
            else min(set(tiers) | {option.tier_index})
        )
        if not self._tier_continuity_preserved(
            existing_tiers=set(tiers),
            new_tier=option.tier_index,
            base_tier=base_tier,
        ):
            return False

        group_rank = self._weight_rank(group)
        for tier, existing_group in tiers.items():
            existing_rank = self._weight_rank(existing_group)
            if group_rank > existing_rank and option.tier_index < tier:
                return False
            if group_rank < existing_rank and option.tier_index > tier:
                return False
        return True

    def _simultaneous_loading_gap_feasible(
        self,
        *,
        option: Stage2PlacementOption,
        group: AllocationGroup,
        selected_options_by_group: List[Tuple[AllocationGroup, Stage2PlacementOption]],
    ) -> bool:
        default_gap = max(0, int(self.config.simultaneous_loading_min_bay_gap or 0))
        if default_gap <= 0:
            return True
        for existing_group, existing_option in selected_options_by_group:
            if not groups_have_simultaneous_loading_conflict(group, existing_group):
                continue
            group_gap = simultaneous_loading_safety_gap_bays(group, default_gap)
            existing_gap = simultaneous_loading_safety_gap_bays(existing_group, default_gap)
            min_gap = max(int(group_gap or 0), int(existing_gap or 0), default_gap)
            if self._bay_gap_between_specs(option.bay_spec, existing_option.bay_spec) < min_gap:
                return False
        return True

    @staticmethod
    def _tier_continuity_preserved(
        *,
        existing_tiers: Set[int],
        new_tier: int,
        base_tier: int,
    ) -> bool:
        planned = set(int(tier) for tier in existing_tiers)
        planned.add(int(new_tier))
        if not planned:
            return True

        lowest = min(planned)
        highest = max(planned)
        if lowest != int(base_tier):
            return False
        return planned == set(range(lowest, highest + 1))

    def _greedy_option_key(
        self,
        option: Stage2PlacementOption,
        selected: List[Stage2PlacementOption],
    ) -> Tuple[int, int, int, float, int, int, int, Tuple[int, int], int]:
        selected_columns = {
            (item.bay_spec, item.stack_index)
            for item in selected
        }
        selected_segments = {item.bay_spec for item in selected}
        column_open = 0 if (option.bay_spec, option.stack_index) in selected_columns else 1
        segment_open = 0 if option.bay_spec in selected_segments else 1
        return (
            segment_open,
            option.segment_occupied_tiers,
            option.column_occupied_tiers,
            option.segment_scarcity_cost,
            column_open,
            option.rank,
            option.stack_index,
            ScipStage2BayAllocator._bay_spec_sort_key(option.bay_spec),
            option.tier_index,
        )

    @classmethod
    def _groups_column_compatible(
        cls,
        group_a: AllocationGroup,
        group_b: AllocationGroup,
    ) -> bool:
        if group_a.size != group_b.size:
            return False
        hard_fields = (
            "business_type",
            "container_type",
            "voyage_id",
            "line_key",
        )
        for field_name in hard_fields:
            value_a = cls._normalise_share_value(getattr(group_a, field_name, None))
            value_b = cls._normalise_share_value(getattr(group_b, field_name, None))
            if cls._values_conflict(value_a, value_b):
                return False

        signature_a = dict(cls._container_attribute_signature(group_a))
        signature_b = dict(cls._container_attribute_signature(group_b))
        attr_a = dict(cls._group_attribute_signature(group_a.group_attributes))
        attr_b = dict(cls._group_attribute_signature(group_b.group_attributes))
        for key in set(signature_a) | set(signature_b):
            if cls._values_conflict(signature_a.get(key), signature_b.get(key)):
                return False
        for key in set(attr_a) | set(attr_b):
            if cls._values_conflict(attr_a.get(key), attr_b.get(key)):
                return False
        return True

    @staticmethod
    def _values_conflict(value_a: Any, value_b: Any) -> bool:
        if value_a in (None, "", (), frozenset()):
            return False
        if value_b in (None, "", (), frozenset()):
            return False
        return value_a != value_b

    def _build_options(
        self,
        area: YardArea,
    ) -> Dict[ContainerSize, List[Stage2PlacementOption]]:
        zone_by_bay = self._crane_zones(area)
        options: Dict[ContainerSize, List[Stage2PlacementOption]] = defaultdict(list)
        existing_20ft_bays = set(getattr(area, "_stage2_existing_20ft_bays", set()))
        existing_large_bays = set(getattr(area, "_stage2_existing_large_bays", set()))

        single_slots: List[Dict[str, int]] = []
        single_capacity_by_bay: Dict[int, int] = defaultdict(int)
        single_fallback_occupied_by_bay: Dict[int, int] = defaultdict(int)
        single_explicit_occupied_by_bay: Dict[int, int] = {}
        for slot in self._single_slots(area):
            bay = int(slot["bay_number"])
            if bay in existing_large_bays:
                continue
            slot_with_tiers = dict(slot)
            tiers = self._slot_tiers(slot_with_tiers, area.max_stack_height)
            if not tiers:
                continue
            slot_with_tiers["tiers"] = tiers
            slot_with_tiers["column_occupied_tiers"] = self._slot_occupied_tiers(
                slot_with_tiers,
                area.max_stack_height,
            )
            explicit_segment_occupied = self._slot_segment_occupied_tiers(slot_with_tiers)
            if explicit_segment_occupied is None:
                single_fallback_occupied_by_bay[bay] += int(
                    slot_with_tiers["column_occupied_tiers"]
                )
            else:
                single_explicit_occupied_by_bay[bay] = max(
                    single_explicit_occupied_by_bay.get(bay, 0),
                    explicit_segment_occupied,
                )
            single_slots.append(slot_with_tiers)
            single_capacity_by_bay[bay] += len(tiers)
        max_single_capacity = max(single_capacity_by_bay.values(), default=1)

        for rank, slot in enumerate(single_slots):
            bay = int(slot["bay_number"])
            stack = int(slot["stack_index"])
            segment_capacity = max(1, single_capacity_by_bay.get(bay, 0))
            segment_occupied_tiers = single_explicit_occupied_by_bay.get(
                bay,
                single_fallback_occupied_by_bay.get(bay, 0),
            )
            for tier in self._slot_tiers(slot, area.max_stack_height):
                options[ContainerSize.SIZE_20].append(
                    Stage2PlacementOption(
                        option_id=f"s20_{bay}_{stack}_{tier}",
                        bay_spec=bay,
                        atoms=((bay, stack, tier),),
                        stack_index=stack,
                        tier_index=tier,
                        zone=zone_by_bay.get(bay, "B"),
                        rank=rank,
                        segment_capacity=segment_capacity,
                        segment_scarcity_cost=max(
                            0.0,
                            float(max_single_capacity - segment_capacity),
                        ),
                        column_occupied_tiers=int(slot.get("column_occupied_tiers") or 0),
                        segment_occupied_tiers=int(segment_occupied_tiers),
                    )
                )

        large_slots: List[Dict[str, Any]] = []
        large_capacity_by_segment: Dict[BaySpec, int] = defaultdict(int)
        large_fallback_occupied_by_segment: Dict[BaySpec, int] = defaultdict(int)
        large_explicit_occupied_by_segment: Dict[BaySpec, int] = {}
        for slot in self._large_slots(area):
            bay_numbers = tuple(int(value) for value in slot["bay_numbers"])
            if set(bay_numbers) & existing_20ft_bays:
                continue
            if set(bay_numbers) & existing_large_bays:
                continue
            display_bays = tuple(int(value) for value in slot.get("display_bays", bay_numbers))
            slot_with_tiers = dict(slot)
            tiers = self._slot_tiers(slot_with_tiers, area.max_stack_height)
            if not tiers:
                continue
            slot_with_tiers["tiers"] = tiers
            slot_with_tiers["column_occupied_tiers"] = self._slot_occupied_tiers(
                slot_with_tiers,
                area.max_stack_height,
            )
            explicit_segment_occupied = self._slot_segment_occupied_tiers(slot_with_tiers)
            if explicit_segment_occupied is None:
                large_fallback_occupied_by_segment[display_bays] += int(
                    slot_with_tiers["column_occupied_tiers"]
                )
            else:
                large_explicit_occupied_by_segment[display_bays] = max(
                    large_explicit_occupied_by_segment.get(display_bays, 0),
                    explicit_segment_occupied,
                )
            large_slots.append(slot_with_tiers)
            large_capacity_by_segment[display_bays] += len(tiers)
        max_large_capacity = max(large_capacity_by_segment.values(), default=1)

        for rank, slot in enumerate(large_slots):
            bay_numbers = tuple(int(value) for value in slot["bay_numbers"])
            display_bays = tuple(int(value) for value in slot.get("display_bays", bay_numbers))
            stack = int(slot["stack_index"])
            zone = self._option_zone(bay_numbers, zone_by_bay)
            segment_capacity = max(1, large_capacity_by_segment.get(display_bays, 0))
            segment_scarcity_cost = max(0.0, float(max_large_capacity - segment_capacity))
            segment_occupied_tiers = large_explicit_occupied_by_segment.get(
                display_bays,
                large_fallback_occupied_by_segment.get(display_bays, 0),
            )
            for tier in self._slot_tiers(slot, area.max_stack_height):
                atoms = tuple((bay, stack, tier) for bay in bay_numbers)
                option = Stage2PlacementOption(
                    option_id=(
                        f"l40_{display_bays[0]}_{display_bays[1]}_"
                        f"{stack}_{tier}_{rank}"
                    ),
                    bay_spec=display_bays,
                    atoms=atoms,
                    stack_index=stack,
                    tier_index=tier,
                    zone=zone,
                    rank=rank,
                    segment_capacity=segment_capacity,
                    segment_scarcity_cost=segment_scarcity_cost,
                    column_occupied_tiers=int(slot.get("column_occupied_tiers") or 0),
                    segment_occupied_tiers=int(segment_occupied_tiers),
                )
                options[ContainerSize.SIZE_40].append(option)
                if bool(slot.get("is_edge_pair")):
                    options[ContainerSize.SIZE_45].append(
                        Stage2PlacementOption(
                            option_id=(
                                f"l45_{display_bays[0]}_{display_bays[1]}_"
                                f"{stack}_{tier}_{rank}"
                            ),
                            bay_spec=display_bays,
                            atoms=atoms,
                            stack_index=stack,
                            tier_index=tier,
                            zone=zone,
                            rank=rank,
                            segment_capacity=segment_capacity,
                            segment_scarcity_cost=segment_scarcity_cost,
                            column_occupied_tiers=int(slot.get("column_occupied_tiers") or 0),
                            segment_occupied_tiers=int(segment_occupied_tiers),
                        )
                    )

        for size in options:
            options[size].sort(
                key=lambda option: (
                    option.rank,
                    option.stack_index,
                    option.tier_index,
                    option.bay_spec,
                )
            )
        return dict(options)

    def _single_slots(self, area: YardArea) -> List[Dict[str, int]]:
        explicit = getattr(area, "_stage2_single_slots", None)
        if explicit is not None:
            return sorted(
                explicit,
                key=lambda slot: (
                    int(slot["bay_number"]),
                    int(slot["stack_index"]),
                    min(self._slot_tiers(slot, area.max_stack_height), default=1),
                ),
            )

        slots: List[Dict[str, int]] = []
        for bay in sorted(area.bays, key=lambda item: item.bay_number):
            if not bay.can_accept_20ft():
                continue
            for stack in range(1, bay.free_columns + 1):
                slots.append(
                    {
                        "bay_number": bay.bay_number,
                        "stack_index": stack,
                        "tiers": list(range(1, area.max_stack_height + 1)),
                        "column_occupied_tiers": 0,
                        "segment_occupied_tiers": int(bay.occupied_columns) * int(area.max_stack_height),
                    }
                )
        return slots

    def _large_slots(self, area: YardArea) -> List[Dict[str, Any]]:
        explicit = getattr(area, "_stage2_large_slots", None)
        if explicit is not None:
            return sorted(
                explicit,
                key=lambda slot: (
                    tuple(int(value) for value in slot["display_bays"]),
                    int(slot["stack_index"]),
                    min(self._slot_tiers(slot, area.max_stack_height), default=1),
                ),
            )

        slots: List[Dict[str, Any]] = []
        for pair in sorted(
            area.large_bay_pairs,
            key=lambda item: (item.bay_a.bay_number, item.bay_b.bay_number),
        ):
            bay_numbers = (pair.bay_a.bay_number, pair.bay_b.bay_number)
            for stack in range(1, pair.free_columns + 1):
                slots.append(
                    {
                        "pair_id": pair.pair_id,
                        "bay_numbers": bay_numbers,
                        "display_bays": bay_numbers,
                        "stack_index": stack,
                        "tiers": list(range(1, area.max_stack_height + 1)),
                        "is_edge_pair": pair.is_edge_pair,
                        "column_occupied_tiers": 0,
                        "segment_occupied_tiers": int(pair.occupied_columns) * int(area.max_stack_height),
                    }
                )
        return slots

    @staticmethod
    def _slot_tiers(slot: Dict[str, Any], max_stack_height: int) -> List[int]:
        raw_tiers = slot.get("tiers")
        if raw_tiers:
            tiers = [
                int(tier)
                for tier in raw_tiers
                if 1 <= int(tier) <= int(max_stack_height)
            ]
            return sorted(set(tiers))

        tier = int(slot.get("tier_index", 1) or 1)
        if tier < 1 or tier > int(max_stack_height):
            return []
        return [tier]

    @classmethod
    def _slot_occupied_tiers(cls, slot: Dict[str, Any], max_stack_height: int) -> int:
        for key in ("column_occupied_tiers", "occupied_tiers", "top_occupied_tier"):
            if key not in slot:
                continue
            try:
                return max(0, int(slot.get(key) or 0))
            except (TypeError, ValueError):
                continue
        tiers = cls._slot_tiers(slot, max_stack_height)
        if not tiers:
            return 0
        return max(0, min(tiers) - 1)

    @staticmethod
    def _slot_segment_occupied_tiers(slot: Dict[str, Any]) -> Optional[int]:
        for key in ("segment_occupied_tiers", "bay_occupied_tiers"):
            if key not in slot:
                continue
            try:
                return max(0, int(slot.get(key) or 0))
            except (TypeError, ValueError):
                continue
        return None

    def _crane_zones(self, area: YardArea) -> Dict[int, str]:
        bay_numbers = sorted(
            {
                int(bay.bay_number)
                for bay in area.bays
            }
            | {
                int(value)
                for slot in self._large_slots(area)
                for value in slot["bay_numbers"]
            }
        )
        if not bay_numbers:
            return {}
        split = int(math.ceil(len(bay_numbers) / 2.0))
        return {
            bay_number: ("A" if index < split else "B")
            for index, bay_number in enumerate(bay_numbers)
        }

    def _option_zone(self, bay_numbers: Iterable[int], zone_by_bay: Dict[int, str]) -> str:
        values = [int(value) for value in bay_numbers]
        if not values:
            return "B"
        zones = [zone_by_bay.get(value, "B") for value in values]
        if zones.count("A") >= zones.count("B"):
            return "A"
        return "B"

    def _build_allocation(
        self,
        area: YardArea,
        assignment: AreaAssignment,
        group: AllocationGroup,
        selected: List[Stage2PlacementOption],
        demand: int,
        unmet_count: int,
        status: str,
    ) -> BayColumnAllocation:
        stacks_by_segment: Dict[BaySpec, Set[int]] = defaultdict(set)
        tiers_by_column: Dict[Tuple[BaySpec, int], List[int]] = defaultdict(list)
        for option in selected:
            stacks_by_segment[option.bay_spec].add(option.stack_index)
            tiers_by_column[(option.bay_spec, option.stack_index)].append(
                option.tier_index
            )

        grouped = {
            bay_spec: len(stacks)
            for bay_spec, stacks in stacks_by_segment.items()
        }

        details = [
            (bay_spec, grouped[bay_spec])
            for bay_spec in sorted(grouped, key=self._bay_spec_sort_key)
        ]
        stack_details = [
            (bay_spec, start_stack, end_stack)
            for bay_spec in sorted(stacks_by_segment, key=self._bay_spec_sort_key)
            for start_stack, end_stack in self._contiguous_ranges(
                list(stacks_by_segment[bay_spec])
            )
        ]
        placed = len(selected)
        exact = "; ".join(
            f"{bay_spec}: stack {stack} tiers {self._format_stack_ranges(tiers)}"
            for (bay_spec, stack), tiers in sorted(
                tiers_by_column.items(),
                key=lambda item: (
                    self._bay_spec_sort_key(item[0][0]),
                    int(item[0][1]),
                ),
            )
        )
        notes = [
            f"greedy stage2 status={status}",
            f"placed={placed}/{demand}",
            f"columns={sum(grouped.values())}",
        ]
        if unmet_count > 0:
            notes.append(f"unmet={unmet_count}")
            logger.warning(
                f"Stage2 greedy area {area.area_id} group {group.group_id} "
                f"placed {placed}/{demand}, unmet {unmet_count}"
            )
        if exact:
            notes.append(f"exact {exact}")

        return BayColumnAllocation(
            allocation_id=f"ALC-{uuid.uuid4().hex[:8].upper()}",
            group_id=group.group_id,
            yard_area_id=assignment.yard_area_id,
            business_type=group.business_type,
            size=group.size,
            split_index=assignment.split_index,
            bay_column_details=details,
            bay_stack_details=stack_details,
            is_edge_placement=group.size == ContainerSize.SIZE_45,
            is_spanning=group.size in (ContainerSize.SIZE_40, ContainerSize.SIZE_45),
            notes="; ".join(notes),
        )

    @staticmethod
    def _item_key(assignment: AreaAssignment, index: int) -> str:
        return (
            f"{index}_{assignment.group_id}_{assignment.split_index}_"
            f"{assignment.assignment_id}"
        ).replace("-", "_")

    @staticmethod
    def _tier_demand(assignment: AreaAssignment, group: AllocationGroup) -> int:
        container_count = max(0, int(group.container_count or len(group.containers) or 0))
        if container_count <= 0:
            return max(0, int(assignment.column_demand))

        group_columns = max(0, int(group.column_demand or 0))
        assigned_columns = max(0, int(assignment.column_demand or 0))
        if group_columns > 0 and assigned_columns > 0 and assigned_columns < group_columns:
            return max(1, int(round(container_count * assigned_columns / group_columns)))
        return container_count

    @classmethod
    def _container_attribute_signature(cls, group: AllocationGroup) -> Tuple[Any, ...]:
        if not group.containers:
            return ()
        fields = (
            "iso_type",
            "category",
            "pod",
            "cattier_kind",
            "trade_code",
            "service_line_code",
            "freight_kind",
            "owner_company",
            "line_company",
            "truck_company",
            "belonger_company",
            "work_type",
            "bol",
            "damage_code",
            "is_reefer",
            "is_hazardous",
            "is_damage",
            "is_high",
            "is_gauge",
            "is_dirty",
        )
        signature: List[Any] = []
        for field_name in fields:
            values = frozenset(
                cls._normalise_share_value(getattr(container, field_name, None))
                for container in group.containers
                if cls._normalise_share_value(getattr(container, field_name, None))
                not in (None, "", (), frozenset())
            )
            if values:
                signature.append((field_name, values))
        return tuple(signature)

    @classmethod
    def _group_attribute_signature(cls, attributes: Dict[str, Any]) -> Tuple[Any, ...]:
        ignored = {
            "weightClass",
            "weight_class",
            "weightMin",
            "weightMax",
            "weight_min",
            "weight_max",
            "filterName",
        }

        def flatten(prefix: str, value: Any) -> List[Tuple[str, Any]]:
            leaf_key = prefix.rsplit(".", 1)[-1]
            if prefix in ignored or leaf_key in ignored:
                return []
            if value in (None, ""):
                return []
            if isinstance(value, dict):
                entries: List[Tuple[str, Any]] = []
                for key in sorted(value):
                    child_prefix = str(key) if not prefix else f"{prefix}.{key}"
                    entries.extend(flatten(child_prefix, value[key]))
                return entries
            normalised = cls._normalise_share_value(value)
            if normalised in (None, "", (), frozenset()):
                return []
            return [(prefix, normalised)]

        entries: List[Tuple[str, Any]] = []
        for key in sorted(attributes):
            entries.extend(flatten(str(key), attributes[key]))
        return tuple(entries)

    @staticmethod
    def _normalise_share_value(value: Any) -> Any:
        if value in (None, ""):
            return None
        if isinstance(value, list):
            return tuple(
                item
                for item in (ScipStage2BayAllocator._normalise_share_value(v) for v in value)
                if item not in (None, "")
            )
        if isinstance(value, set):
            return frozenset(
                item
                for item in (ScipStage2BayAllocator._normalise_share_value(v) for v in value)
                if item not in (None, "")
            )
        if hasattr(value, "value"):
            return value.value
        return value

    @staticmethod
    def _weight_rank(group: AllocationGroup) -> int:
        value = group.weight_class
        raw = value.value if hasattr(value, "value") else value
        return {
            "empty": 0,
            "light": 1,
            "heavy": 2,
            0: 0,
            1: 1,
            2: 2,
        }.get(raw, 1)

    @staticmethod
    def _segment_name(segment: BaySpec) -> str:
        if isinstance(segment, tuple):
            return "_".join(str(value) for value in segment)
        return str(segment)

    @staticmethod
    def _bay_spec_sort_key(segment: BaySpec) -> Tuple[int, int]:
        if isinstance(segment, tuple):
            values = [int(value) for value in segment]
            return (min(values), max(values))
        value = int(segment)
        return (value, value)

    @staticmethod
    def _bay_gap_between_specs(left: BaySpec, right: BaySpec) -> int:
        left_min, left_max = ScipStage2BayAllocator._bay_spec_sort_key(left)
        right_min, right_max = ScipStage2BayAllocator._bay_spec_sort_key(right)
        if left_max < right_min:
            return right_min - left_max
        if right_max < left_min:
            return left_min - right_max
        return 0

    @staticmethod
    def _contiguous_ranges(stacks: List[int]) -> List[Tuple[int, int]]:
        values = sorted({int(stack) for stack in stacks})
        if not values:
            return []

        ranges: List[Tuple[int, int]] = []
        start = previous = values[0]
        for value in values[1:]:
            if value == previous + 1:
                previous = value
                continue
            ranges.append((start, previous))
            start = previous = value
        ranges.append((start, previous))
        return ranges

    @classmethod
    def _format_stack_ranges(cls, stacks: List[int]) -> str:
        parts = []
        for start, end in cls._contiguous_ranges(stacks):
            parts.append(str(start) if start == end else f"{start}-{end}")
        return ",".join(parts)
