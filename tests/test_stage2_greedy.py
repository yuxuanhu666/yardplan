import unittest

from yardplan_core.models import (
    AllocationGroup,
    AreaAssignment,
    Bay,
    BusinessType,
    ContainerSize,
    ContainerType,
    LargeBayPair,
    WeightClass,
    YardArea,
)
from yardplan_core.stage2_scip import (
    ScipStage2BayAllocator,
    _bays_from_bay_spec,
)
from yardplan_core.visualization import (
    PlannedDrawItem,
    StackCell,
    _cell_conflicts_with_item,
    _items_from_bay_allocations,
    _merge_plan_label_region,
)


class CapturingAllocator(ScipStage2BayAllocator):
    def __init__(self):
        super().__init__()
        self.selected_by_group = {}

    def _build_allocation(self, *args, **kwargs):
        group = kwargs["group"]
        selected = kwargs["selected"]
        self.selected_by_group[group.group_id] = list(selected)
        return super()._build_allocation(*args, **kwargs)


def make_group(
    group_id: str,
    size: ContainerSize,
    demand: int,
    *,
    weight_class: WeightClass = WeightClass.LIGHT,
    voyage_id: str = "V001",
    line_key: int = 1,
) -> AllocationGroup:
    return AllocationGroup(
        group_id=group_id,
        business_type=BusinessType.IMPORT,
        size=size,
        container_type=ContainerType.DRY,
        weight_class=weight_class,
        voyage_id=voyage_id,
        line_key=line_key,
        column_demand=demand,
        container_count=demand,
    )


def make_assignment(group_id: str, demand: int, area_id: str = "A") -> AreaAssignment:
    return AreaAssignment(
        assignment_id=f"ASN-{group_id}",
        group_id=group_id,
        yard_area_id=area_id,
        column_demand=demand,
    )


def make_area(
    *,
    bay_count: int,
    columns: int,
    height: int,
    edge_pairs=None,
) -> YardArea:
    edge_pairs = set(edge_pairs or [])
    bays = [
        Bay(
            bay_id=f"A-{bay_number}",
            bay_number=bay_number,
            yard_area_id="A",
            total_columns=columns,
        )
        for bay_number in range(1, bay_count + 1)
    ]
    by_number = {bay.bay_number: bay for bay in bays}
    pairs = []
    for left in range(1, bay_count, 2):
        right = left + 1
        pairs.append(
            LargeBayPair(
                pair_id=f"A-{left}_{right}",
                yard_area_id="A",
                bay_a=by_number[left],
                bay_b=by_number[right],
                is_edge_pair=(left, right) in edge_pairs,
            )
        )
    return YardArea(
        area_id="A",
        business_type=BusinessType.IMPORT,
        bays=bays,
        large_bay_pairs=pairs,
        max_stack_height=height,
    )


def assert_hard_constraints(testcase, allocator, groups_by_id):
    occupied_atoms = set()
    used_20ft_bays = set()
    used_large_bays = set()
    column_entries = {}

    for group_id, selected in allocator.selected_by_group.items():
        group = groups_by_id[group_id]
        for option in selected:
            for atom in option.atoms:
                testcase.assertNotIn(atom, occupied_atoms)
                occupied_atoms.add(atom)

            bays = set(_bays_from_bay_spec(option.bay_spec))
            if group.size == ContainerSize.SIZE_20:
                testcase.assertFalse(bays & used_large_bays)
                used_20ft_bays.update(bays)
            else:
                testcase.assertFalse(bays & used_20ft_bays)
                used_large_bays.update(bays)

            column_key = (option.bay_spec, option.stack_index)
            column_entries.setdefault(column_key, []).append((option.tier_index, group))

    for (_bay_spec, _stack), entries in column_entries.items():
        sizes = {group.size for _tier, group in entries}
        testcase.assertLessEqual(len(sizes), 1)

        tiers = sorted(tier for tier, _group in entries)
        testcase.assertEqual(tiers, list(range(tiers[0], tiers[-1] + 1)))

        for index, (tier_a, group_a) in enumerate(entries):
            for tier_b, group_b in entries[index + 1 :]:
                testcase.assertTrue(
                    allocator._groups_column_compatible(group_a, group_b)
                    or group_a.group_id == group_b.group_id
                )
                rank_a = allocator._weight_rank(group_a)
                rank_b = allocator._weight_rank(group_b)
                if rank_a > rank_b:
                    testcase.assertGreater(tier_a, tier_b)
                if rank_b > rank_a:
                    testcase.assertGreater(tier_b, tier_a)


def selected_atoms(allocator):
    atoms = []
    for selected in allocator.selected_by_group.values():
        for option in selected:
            atoms.extend(option.atoms)
    return atoms


class Stage2GreedyTests(unittest.TestCase):
    def test_greedy_places_45_on_edge_and_locks_bay_types(self):
        area = make_area(
            bay_count=6,
            columns=2,
            height=2,
            edge_pairs={(1, 2), (5, 6)},
        )
        groups = [
            make_group("G45", ContainerSize.SIZE_45, 2),
            make_group("G40", ContainerSize.SIZE_40, 2),
            make_group("G20", ContainerSize.SIZE_20, 2),
        ]
        assignments = [make_assignment(group.group_id, group.container_count) for group in groups]
        allocator = CapturingAllocator()

        allocations = allocator.allocate(
            assignments,
            {group.group_id: group for group in groups},
            {area.area_id: area},
        )

        self.assertEqual(len(allocations), 3)
        self.assertEqual(len(allocator.selected_by_group["G45"]), 2)
        self.assertEqual(len(allocator.selected_by_group["G40"]), 2)
        self.assertEqual(len(allocator.selected_by_group["G20"]), 2)
        self.assertTrue(
            all(
                option.bay_spec in {(1, 2), (5, 6)}
                for option in allocator.selected_by_group["G45"]
            )
        )
        assert_hard_constraints(self, allocator, {group.group_id: group for group in groups})

    def test_shared_column_allows_weight_difference_only(self):
        area = make_area(bay_count=1, columns=1, height=2)
        light = make_group("LIGHT", ContainerSize.SIZE_20, 1, weight_class=WeightClass.LIGHT)
        heavy = make_group("HEAVY", ContainerSize.SIZE_20, 1, weight_class=WeightClass.HEAVY)
        groups = [heavy, light]
        allocator = CapturingAllocator()

        allocator.allocate(
            [make_assignment(group.group_id, group.container_count) for group in groups],
            {group.group_id: group for group in groups},
            {area.area_id: area},
        )

        light_option = allocator.selected_by_group["LIGHT"][0]
        heavy_option = allocator.selected_by_group["HEAVY"][0]
        self.assertEqual((light_option.bay_spec, light_option.stack_index), (1, 1))
        self.assertEqual((heavy_option.bay_spec, heavy_option.stack_index), (1, 1))
        self.assertLess(light_option.tier_index, heavy_option.tier_index)
        assert_hard_constraints(self, allocator, {group.group_id: group for group in groups})

    def test_conflicting_groups_do_not_share_physical_column(self):
        area = make_area(bay_count=1, columns=2, height=2)
        first = make_group("A1", ContainerSize.SIZE_20, 1, voyage_id="V001")
        second = make_group("A2", ContainerSize.SIZE_20, 1, voyage_id="V002")
        groups = [first, second]
        allocator = CapturingAllocator()

        allocator.allocate(
            [make_assignment(group.group_id, group.container_count) for group in groups],
            {group.group_id: group for group in groups},
            {area.area_id: area},
        )

        first_option = allocator.selected_by_group["A1"][0]
        second_option = allocator.selected_by_group["A2"][0]
        self.assertNotEqual(
            (first_option.bay_spec, first_option.stack_index),
            (second_option.bay_spec, second_option.stack_index),
        )
        assert_hard_constraints(self, allocator, {group.group_id: group for group in groups})

    def test_discontinuous_tier_candidates_are_not_used(self):
        area = make_area(bay_count=1, columns=1, height=3)
        area._stage2_single_slots = [
            {
                "bay_number": 1,
                "stack_index": 1,
                "tiers": [1, 3],
            }
        ]
        group = make_group("G20", ContainerSize.SIZE_20, 2)
        allocator = CapturingAllocator()

        allocations = allocator.allocate(
            [make_assignment(group.group_id, group.container_count)],
            {group.group_id: group},
            {area.area_id: area},
        )

        selected = allocator.selected_by_group["G20"]
        self.assertEqual([option.tier_index for option in selected], [1])
        self.assertIn("unmet=1", allocations[0].notes)
        assert_hard_constraints(self, allocator, {group.group_id: group})

    def test_selected_atoms_are_globally_unique(self):
        area = make_area(
            bay_count=4,
            columns=3,
            height=3,
            edge_pairs={(1, 2), (3, 4)},
        )
        groups = [
            make_group("G45", ContainerSize.SIZE_45, 3),
            make_group("G40", ContainerSize.SIZE_40, 3),
            make_group("G20A", ContainerSize.SIZE_20, 3, voyage_id="V20A"),
            make_group("G20B", ContainerSize.SIZE_20, 3, voyage_id="V20B"),
        ]
        allocator = CapturingAllocator()

        allocator.allocate(
            [make_assignment(group.group_id, group.container_count) for group in groups],
            {group.group_id: group for group in groups},
            {area.area_id: area},
        )

        atoms = selected_atoms(allocator)
        self.assertEqual(len(atoms), len(set(atoms)))
        assert_hard_constraints(self, allocator, {group.group_id: group for group in groups})

    def test_visualization_conflict_uses_exact_tiers_when_available(self):
        area = make_area(bay_count=1, columns=1, height=3)
        area._stage2_single_slots = [
            {
                "bay_number": 1,
                "stack_index": 1,
                "tiers": [3],
            }
        ]
        group = make_group("G20", ContainerSize.SIZE_20, 1)
        allocator = CapturingAllocator()
        allocations = allocator.allocate(
            [make_assignment(group.group_id, group.container_count)],
            {group.group_id: group},
            {area.area_id: area},
        )

        draw_items = _items_from_bay_allocations(allocations, None)
        self.assertEqual(draw_items[0].tier_ranges, ((3, 3),))

        occupied_below = StackCell(
            block_id="A",
            bay_idx=1,
            stack_idx=1,
            x=0,
            y=0,
            width=1,
            height=1,
            occupied=True,
            occupied_tiers=(1, 2),
        )
        occupied_same_tier = StackCell(
            block_id="A",
            bay_idx=1,
            stack_idx=1,
            x=0,
            y=0,
            width=1,
            height=1,
            occupied=True,
            occupied_tiers=(3,),
        )
        self.assertFalse(_cell_conflicts_with_item(occupied_below, draw_items[0]))
        self.assertTrue(_cell_conflicts_with_item(occupied_same_tier, draw_items[0]))

    def test_visualization_labels_merge_once_per_bay_and_group(self):
        regions = {}
        first_stack = PlannedDrawItem(
            group_id="G20",
            block_id="A",
            bay_start=1,
            bay_end=1,
            stack_start=1,
            stack_end=1,
        )
        second_stack = PlannedDrawItem(
            group_id="G20",
            block_id="A",
            bay_start=1,
            bay_end=1,
            stack_start=2,
            stack_end=2,
        )
        other_group = PlannedDrawItem(
            group_id="H20",
            block_id="A",
            bay_start=1,
            bay_end=1,
            stack_start=3,
            stack_end=3,
        )
        other_bay = PlannedDrawItem(
            group_id="G20",
            block_id="A",
            bay_start=2,
            bay_end=2,
            stack_start=1,
            stack_end=1,
        )

        _merge_plan_label_region(regions, first_stack, (0.0, 0.0, 1.0, 1.0))
        _merge_plan_label_region(regions, second_stack, (1.0, 0.0, 2.0, 1.0))
        _merge_plan_label_region(regions, other_group, (2.0, 0.0, 3.0, 1.0))
        _merge_plan_label_region(regions, other_bay, (0.0, 1.0, 1.0, 2.0))

        self.assertEqual(len(regions), 3)
        self.assertEqual(regions[("A", 1, 1, "G20")], (0.0, 0.0, 2.0, 1.0))
        self.assertEqual(regions[("A", 1, 1, "H20")], (2.0, 0.0, 3.0, 1.0))
        self.assertEqual(regions[("A", 2, 2, "G20")], (0.0, 1.0, 1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
