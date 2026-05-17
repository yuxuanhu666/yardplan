from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from yardplan_core.models import (
    AllocationGroup,
    AreaAssignment,
    BayColumnAllocation,
    ContainerSize,
    YardArea,
    logger,
)

Atom = Tuple[int, int]
BaySpec = Any


@dataclass(frozen=True)
class Stage2PlacementOption:
    """One feasible column placement candidate inside a yard area."""

    option_id: str
    bay_spec: BaySpec
    atoms: Tuple[Atom, ...]
    stack_index: int
    zone: str
    rank: int


@dataclass
class Stage2ScipConfig:
    time_limit_seconds: float = 20.0
    unmet_weight: float = 1_000_000.0
    segment_weight: float = 1_000.0
    region_component_weight: float = 250.0
    enforce_stack_contiguity: bool = True
    segment_min_fill_ratio: float = 0.65
    max_underfilled_segments_per_item: int = 2
    crane_balance_weight: float = 10.0
    stability_weight: float = 0.001


class ScipStage2BayAllocator:
    """
    SCIP implementation of stage 2 bay/column allocation.

    The model is built independently per yard area. Every decision consumes
    bottom-level atoms `(bay_number, stack_index)`, so 20/40/45ft placements
    share the same physical resources and cannot overlap.
    """

    def __init__(self, config: Optional[Stage2ScipConfig] = None):
        self.config = config or Stage2ScipConfig()

    def allocate(
        self,
        area_assignments: List[AreaAssignment],
        groups: Dict[str, AllocationGroup],
        yard_areas: Dict[str, YardArea],
    ) -> List[BayColumnAllocation]:
        by_area: Dict[str, List[Tuple[AreaAssignment, AllocationGroup]]] = defaultdict(list)
        for assignment in area_assignments:
            group = groups.get(assignment.group_id)
            if group is None or assignment.yard_area_id not in yard_areas:
                continue
            by_area[assignment.yard_area_id].append((assignment, group))

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
        try:
            from pyscipopt import Model, quicksum
        except ImportError as exc:
            raise RuntimeError(
                "SCIP 第二阶段需要安装 PySCIPOpt，并且系统需可找到 SCIP C 库/头文件。"
                "请先安装 SCIP，再安装 PySCIPOpt。"
            ) from exc

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
        item_keys = [self._item_key(assignment, index) for index, (assignment, _group) in enumerate(items)]
        options_by_size = self._build_options(area)

        model = Model(f"stage2_{area.area_id}")
        model.hideOutput(True)
        if self.config.time_limit_seconds > 0:
            model.setRealParam("limits/time", float(self.config.time_limit_seconds))

        x: Dict[Tuple[str, str], Any] = {}
        y: Dict[Tuple[str, BaySpec], Any] = {}
        unmet: Dict[str, Any] = {}
        placed_expr: Dict[str, Any] = {}
        option_lookup: Dict[str, Stage2PlacementOption] = {}

        for option_list in options_by_size.values():
            for option in option_list:
                option_lookup[option.option_id] = option

        atom_to_vars: Dict[Atom, List[Any]] = defaultdict(list)
        bay_to_vars: Dict[int, List[Any]] = defaultdict(list)
        bay_available_atoms = self._available_atoms_by_bay(options_by_size)
        used_20ft_bays: Dict[int, Any] = {}
        used_large_pairs: Dict[BaySpec, Any] = {}
        large_pairs_by_bay: Dict[int, List[Any]] = defaultdict(list)
        segment_stack_vars: Dict[Tuple[str, BaySpec], Dict[int, Any]] = defaultdict(dict)
        item_segments: Dict[str, Dict[BaySpec, Any]] = defaultdict(dict)

        for item_key, (assignment, group) in zip(item_keys, items):
            demand = max(0, int(assignment.column_demand))
            size_options = options_by_size.get(group.size, [])
            unmet[item_key] = model.addVar(
                vtype="I",
                lb=0,
                ub=demand,
                name=f"unmet_{item_key}",
            )

            item_vars: List[Any] = []
            for option in size_options:
                var = model.addVar(vtype="B", name=f"x_{item_key}_{option.option_id}")
                x[(item_key, option.option_id)] = var
                item_vars.append(var)

                segment_key = option.bay_spec
                if (item_key, segment_key) not in y:
                    y[(item_key, segment_key)] = model.addVar(
                        vtype="B",
                        name=f"seg_{item_key}_{self._segment_name(segment_key)}",
                    )
                    item_segments[item_key][segment_key] = y[(item_key, segment_key)]
                model.addCons(var <= y[(item_key, segment_key)])
                segment_stack_vars[(item_key, segment_key)][option.stack_index] = var

                for atom in option.atoms:
                    atom_to_vars[atom].append(var)
                    bay_to_vars[atom[0]].append(var)

                if group.size == ContainerSize.SIZE_20:
                    bay_number = option.atoms[0][0]
                    if bay_number not in used_20ft_bays:
                        used_20ft_bays[bay_number] = model.addVar(
                            vtype="B",
                            name=f"use20_bay_{bay_number}",
                        )
                    model.addCons(var <= used_20ft_bays[bay_number])
                else:
                    if option.bay_spec not in used_large_pairs:
                        pair_var = model.addVar(
                            vtype="B",
                            name=f"use_large_{self._segment_name(option.bay_spec)}",
                        )
                        used_large_pairs[option.bay_spec] = pair_var
                        for bay_number in {atom[0] for atom in option.atoms}:
                            large_pairs_by_bay[bay_number].append(pair_var)
                    model.addCons(var <= used_large_pairs[option.bay_spec])

            placed = quicksum(item_vars) if item_vars else 0
            placed_expr[item_key] = placed
            model.addCons(placed + unmet[item_key] == demand, name=f"demand_{item_key}")

        region_start_terms: List[Any] = []
        if self.config.enforce_stack_contiguity:
            self._add_stack_contiguity_constraints(model, segment_stack_vars)
        self._add_segment_fill_constraints(
            model,
            segment_stack_vars,
            item_segments,
            item_keys,
            region_start_terms,
        )

        for atom, atom_vars in sorted(atom_to_vars.items()):
            model.addCons(quicksum(atom_vars) <= 1, name=f"atom_{atom[0]}_{atom[1]}")

        for bay_number, bay_vars in sorted(bay_to_vars.items()):
            capacity = len(bay_available_atoms.get(bay_number, set()))
            model.addCons(
                quicksum(bay_vars) <= capacity,
                name=f"bay_capacity_{bay_number}",
            )

        all_bays = set(used_20ft_bays) | set(large_pairs_by_bay)
        for bay_number in sorted(all_bays):
            terms: List[Any] = []
            if bay_number in used_20ft_bays:
                terms.append(used_20ft_bays[bay_number])
            terms.extend(large_pairs_by_bay.get(bay_number, []))
            if terms:
                model.addCons(
                    quicksum(terms) <= 1,
                    name=f"bay_type_lock_{bay_number}",
                )

        load_a_terms: List[Any] = []
        load_b_terms: List[Any] = []
        stability_terms: List[Any] = []
        for (item_key, option_id), var in x.items():
            option = option_lookup[option_id]
            if option.zone == "A":
                load_a_terms.append(var)
            else:
                load_b_terms.append(var)
            stability_terms.append(float(option.rank) * var)

        load_a = quicksum(load_a_terms) if load_a_terms else 0
        load_b = quicksum(load_b_terms) if load_b_terms else 0
        balance = model.addVar(vtype="C", lb=0, name="crane_balance_abs")
        model.addCons(balance >= load_a - load_b, name="balance_pos")
        model.addCons(balance >= load_b - load_a, name="balance_neg")

        objective = (
            self.config.unmet_weight * quicksum(unmet.values())
            + self.config.segment_weight * quicksum(y.values())
            + self.config.region_component_weight * quicksum(region_start_terms)
            + self.config.crane_balance_weight * balance
            + self.config.stability_weight * quicksum(stability_terms)
        )
        model.setObjective(objective, "minimize")
        model.optimize()

        status = str(model.getStatus())
        if status not in {"optimal", "timelimit", "gaplimit", "bestsollimit"}:
            logger.warning(f"Stage2 SCIP area {area.area_id} ended with status {status}")

        allocations: List[BayColumnAllocation] = []
        for item_key, (assignment, group) in zip(item_keys, items):
            selected: List[Stage2PlacementOption] = []
            for option in options_by_size.get(group.size, []):
                var = x.get((item_key, option.option_id))
                if var is not None and model.getVal(var) > 0.5:
                    selected.append(option)

            allocation = self._build_allocation(
                area=area,
                assignment=assignment,
                group=group,
                selected=selected,
                demand=max(0, int(assignment.column_demand)),
                unmet_count=int(round(model.getVal(unmet[item_key]))),
                status=status,
            )
            allocations.append(allocation)

        return allocations

    def _add_stack_contiguity_constraints(
        self,
        model: Any,
        segment_stack_vars: Dict[Tuple[str, BaySpec], Dict[int, Any]],
    ) -> None:
        """
        For one item in one bay/large-bay segment, selected stacks must form
        a physical contiguous interval. This prevents shapes like stack 1 and
        stack 9 with holes in between.
        """
        for (item_key, segment), stack_vars in segment_stack_vars.items():
            stacks = sorted(int(stack) for stack in stack_vars)
            if len(stacks) <= 2:
                continue
            segment_name = self._segment_name(segment)
            for left_index, left_stack in enumerate(stacks[:-2]):
                left_var = stack_vars[left_stack]
                for right_stack in stacks[left_index + 2 :]:
                    right_var = stack_vars[right_stack]
                    for middle_stack in range(left_stack + 1, right_stack):
                        middle_var = stack_vars.get(middle_stack)
                        if middle_var is None:
                            model.addCons(
                                left_var + right_var <= 1,
                                name=(
                                    f"stack_no_missing_gap_{item_key}_"
                                    f"{segment_name}_{left_stack}_{right_stack}"
                                ),
                            )
                            break
                        model.addCons(
                            left_var + right_var - 1 <= middle_var,
                            name=(
                                f"stack_no_gap_{item_key}_{segment_name}_"
                                f"{left_stack}_{middle_stack}_{right_stack}"
                            ),
                        )

    def _add_segment_fill_constraints(
        self,
        model: Any,
        segment_stack_vars: Dict[Tuple[str, BaySpec], Dict[int, Any]],
        item_segments: Dict[str, Dict[BaySpec, Any]],
        item_keys: List[str],
        region_start_terms: List[Any],
    ) -> None:
        """
        Encourage each used bay/large-bay segment to be a real compact block.

        Most used segments must reach a minimum fill depth; only a small number
        of tail segments may be underfilled. Region starts are penalized so an
        item can have several blocks, but unnecessary bay-axis fragmentation is
        discouraged.
        """
        min_fill_ratio = max(0.0, min(1.0, self.config.segment_min_fill_ratio))
        max_underfilled = max(1, int(self.config.max_underfilled_segments_per_item))

        for item_key in item_keys:
            segments = item_segments.get(item_key, {})
            if not segments:
                continue

            underfilled_terms: List[Any] = []
            for segment, segment_var in segments.items():
                stack_vars = segment_stack_vars.get((item_key, segment), {})
                if not stack_vars:
                    continue
                capacity = len(stack_vars)
                threshold = max(1, min(capacity, int(math.ceil(capacity * min_fill_ratio))))
                underfilled = model.addVar(
                    vtype="B",
                    name=f"underfill_{item_key}_{self._segment_name(segment)}",
                )
                fill_expr = sum(stack_vars.values())
                model.addCons(
                    fill_expr + threshold * underfilled >= threshold * segment_var,
                    name=f"segment_min_fill_{item_key}_{self._segment_name(segment)}",
                )
                model.addCons(
                    underfilled <= segment_var,
                    name=f"underfill_active_{item_key}_{self._segment_name(segment)}",
                )
                underfilled_terms.append(underfilled)

            if underfilled_terms:
                model.addCons(
                    sum(underfilled_terms) <= max_underfilled,
                    name=f"underfill_limit_{item_key}",
                )

            previous_segment_var: Optional[Any] = None
            for segment in sorted(segments, key=self._bay_spec_sort_key):
                segment_var = segments[segment]
                start_var = model.addVar(
                    vtype="B",
                    name=f"region_start_{item_key}_{self._segment_name(segment)}",
                )
                if previous_segment_var is None:
                    model.addCons(
                        start_var >= segment_var,
                        name=f"region_first_start_{item_key}_{self._segment_name(segment)}",
                    )
                else:
                    model.addCons(
                        start_var >= segment_var - previous_segment_var,
                        name=f"region_start_link_{item_key}_{self._segment_name(segment)}",
                    )
                model.addCons(
                    start_var <= segment_var,
                    name=f"region_start_active_{item_key}_{self._segment_name(segment)}",
                )
                region_start_terms.append(start_var)
                previous_segment_var = segment_var

    def _build_options(
        self,
        area: YardArea,
    ) -> Dict[ContainerSize, List[Stage2PlacementOption]]:
        zone_by_bay = self._crane_zones(area)
        options: Dict[ContainerSize, List[Stage2PlacementOption]] = defaultdict(list)
        existing_20ft_bays = set(getattr(area, "_stage2_existing_20ft_bays", set()))
        existing_large_bays = set(getattr(area, "_stage2_existing_large_bays", set()))

        for rank, slot in enumerate(self._single_slots(area)):
            bay = int(slot["bay_number"])
            if bay in existing_large_bays:
                continue
            stack = int(slot["stack_index"])
            options[ContainerSize.SIZE_20].append(
                Stage2PlacementOption(
                    option_id=f"s20_{bay}_{stack}",
                    bay_spec=bay,
                    atoms=((bay, stack),),
                    stack_index=stack,
                    zone=zone_by_bay.get(bay, "B"),
                    rank=rank,
                )
            )

        large_slots = self._large_slots(area)
        for rank, slot in enumerate(large_slots):
            bay_numbers = tuple(int(value) for value in slot["bay_numbers"])
            if set(bay_numbers) & existing_20ft_bays:
                continue
            if set(bay_numbers) & existing_large_bays:
                continue
            display_bays = tuple(int(value) for value in slot.get("display_bays", bay_numbers))
            stack = int(slot["stack_index"])
            atoms = tuple((bay, stack) for bay in bay_numbers)
            zone = self._option_zone(bay_numbers, zone_by_bay)
            option = Stage2PlacementOption(
                option_id=f"l40_{display_bays[0]}_{display_bays[1]}_{stack}_{rank}",
                bay_spec=display_bays,
                atoms=atoms,
                stack_index=stack,
                zone=zone,
                rank=rank,
            )
            options[ContainerSize.SIZE_40].append(option)
            if bool(slot.get("is_edge_pair")):
                options[ContainerSize.SIZE_45].append(
                    Stage2PlacementOption(
                        option_id=f"l45_{display_bays[0]}_{display_bays[1]}_{stack}_{rank}",
                        bay_spec=display_bays,
                        atoms=atoms,
                        stack_index=stack,
                        zone=zone,
                        rank=rank,
                    )
                )

        for size in options:
            options[size].sort(key=lambda option: (option.rank, option.stack_index, option.bay_spec))
        return dict(options)

    def _single_slots(self, area: YardArea) -> List[Dict[str, int]]:
        explicit = getattr(area, "_stage2_single_slots", None)
        if explicit is not None:
            return sorted(
                explicit,
                key=lambda slot: (int(slot["bay_number"]), int(slot["stack_index"])),
            )

        slots: List[Dict[str, int]] = []
        for bay in sorted(area.bays, key=lambda item: item.bay_number):
            if not bay.can_accept_20ft():
                continue
            for stack in range(1, bay.free_columns + 1):
                slots.append({"bay_number": bay.bay_number, "stack_index": stack})
        return slots

    def _large_slots(self, area: YardArea) -> List[Dict[str, Any]]:
        explicit = getattr(area, "_stage2_large_slots", None)
        if explicit is not None:
            return sorted(
                explicit,
                key=lambda slot: (
                    tuple(int(value) for value in slot["display_bays"]),
                    int(slot["stack_index"]),
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
                        "is_edge_pair": pair.is_edge_pair,
                    }
                )
        return slots

    def _available_atoms_by_bay(
        self,
        options_by_size: Dict[ContainerSize, List[Stage2PlacementOption]],
    ) -> Dict[int, set]:
        atoms_by_bay: Dict[int, set] = defaultdict(set)
        for option in options_by_size.get(ContainerSize.SIZE_20, []):
            for atom in option.atoms:
                atoms_by_bay[atom[0]].add(atom)
        for option in options_by_size.get(ContainerSize.SIZE_40, []):
            for atom in option.atoms:
                atoms_by_bay[atom[0]].add(atom)
        return atoms_by_bay

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
        grouped: Dict[BaySpec, int] = defaultdict(int)
        stacks_by_segment: Dict[BaySpec, List[int]] = defaultdict(list)
        for option in selected:
            grouped[option.bay_spec] += 1
            stacks_by_segment[option.bay_spec].append(option.stack_index)

        details = [
            (bay_spec, grouped[bay_spec])
            for bay_spec in sorted(grouped, key=self._bay_spec_sort_key)
        ]
        stack_details = [
            (bay_spec, start_stack, end_stack)
            for bay_spec in sorted(stacks_by_segment, key=self._bay_spec_sort_key)
            for start_stack, end_stack in self._contiguous_ranges(
                stacks_by_segment[bay_spec]
            )
        ]
        placed = sum(grouped.values())
        exact = "; ".join(
            f"{bay_spec}: stacks {self._format_stack_ranges(stacks)}"
            for bay_spec, stacks in sorted(stacks_by_segment.items(), key=lambda item: self._bay_spec_sort_key(item[0]))
            if stacks
        )
        notes = [
            f"SCIP stage2 status={status}",
            f"placed={placed}/{demand}",
        ]
        if unmet_count > 0:
            notes.append(f"unmet={unmet_count}")
            logger.warning(
                f"Stage2 SCIP area {area.area_id} group {group.group_id} "
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
