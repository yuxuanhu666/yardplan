import unittest
from datetime import datetime

from yardplan_core.models import (
    AllocationGroup,
    BayColumnAllocation,
    BusinessType,
    ContainerSize,
    ContainerType,
    PlanningResult,
    Vessel,
    WeightClass,
)
from yardplan_core.simultaneous import add_simultaneous_loading_conflict
from yardplan_score import score_yard_plan


def make_group(
    group_id: str,
    voyage_id: str,
    container_count: int,
    *,
    weight_class: WeightClass = WeightClass.LIGHT,
) -> AllocationGroup:
    return AllocationGroup(
        group_id=group_id,
        business_type=BusinessType.EXPORT,
        size=ContainerSize.SIZE_20,
        container_type=ContainerType.DRY,
        weight_class=weight_class,
        voyage_id=voyage_id,
        line_key=1,
        container_count=container_count,
        column_demand=container_count,
    )


def make_group_with_business(
    group_id: str,
    voyage_id: str,
    business_type: BusinessType,
    container_count: int,
) -> AllocationGroup:
    return AllocationGroup(
        group_id=group_id,
        business_type=business_type,
        size=ContainerSize.SIZE_20,
        container_type=ContainerType.DRY,
        weight_class=WeightClass.LIGHT,
        voyage_id=voyage_id,
        line_key=1,
        container_count=container_count,
        column_demand=container_count,
    )


def make_allocation(group_id: str, voyage_area_id: str, bay_number: int) -> BayColumnAllocation:
    return BayColumnAllocation(
        allocation_id=f"ALLOC-{group_id}-{voyage_area_id}-{bay_number}",
        group_id=group_id,
        yard_area_id=voyage_area_id,
        business_type=BusinessType.EXPORT,
        size=ContainerSize.SIZE_20,
        bay_column_details=[(bay_number, 1)],
    )


class YardplanScoreTests(unittest.TestCase):
    def test_area_peak_staggering_ignores_same_voyage_import_export_overlap(self):
        groups = [
            make_group_with_business("GI", "V001", BusinessType.IMPORT, 10),
            make_group_with_business("GE", "V001", BusinessType.EXPORT, 10),
        ]
        allocations = [
            make_allocation("GI", "A", 1),
            make_allocation("GE", "A", 2),
        ]
        result = PlanningResult(
            run_id="RUN-AREA-1",
            timestamp=datetime(2026, 1, 1, 0, 0, 0),
            mode=None,
            allocation_groups=groups,
            bay_column_allocations=allocations,
        )

        score = score_yard_plan(result)
        item = score["items"]["area_peak_staggering"]

        self.assertEqual(item["score"], 10.0)
        self.assertEqual(item["details"]["areas"]["A"]["voyageCount"], 1)
        self.assertEqual(item["details"]["areas"]["A"]["pairs"], [])

    def test_area_peak_staggering_penalizes_different_voyage_opposite_direction_overlap(self):
        groups = [
            make_group_with_business("GI", "V001", BusinessType.IMPORT, 10),
            make_group_with_business("GE", "V002", BusinessType.EXPORT, 10),
        ]
        allocations = [
            make_allocation("GI", "A", 1),
            make_allocation("GE", "A", 2),
        ]
        result = PlanningResult(
            run_id="RUN-AREA-2",
            timestamp=datetime(2026, 1, 1, 0, 0, 0),
            mode=None,
            allocation_groups=groups,
            bay_column_allocations=allocations,
        )

        score = score_yard_plan(result)
        item = score["items"]["area_peak_staggering"]
        area_detail = item["details"]["areas"]["A"]

        self.assertEqual(item["score"], 0.0)
        self.assertEqual(area_detail["voyageCount"], 2)
        self.assertEqual(area_detail["conflictRatio"], 1.0)
        self.assertEqual(area_detail["pairs"][0]["score"], 0.0)

    def test_business_dispersion_counts_areas_per_voyage_not_per_root_group(self):
        voyage_id = "V001"
        groups = [
            make_group("GA", voyage_id, 10),
            make_group("GB", voyage_id, 10),
        ]
        allocations = [
            make_allocation("GA", "A", 1),
            make_allocation("GB", "B", 2),
        ]
        vessel = Vessel(
            vessel_id="VESSEL-1",
            vessel_name="Test Vessel",
            voyage_id=voyage_id,
            eta=datetime(2026, 1, 1, 8, 0, 0),
            etd=datetime(2026, 1, 2, 8, 0, 0),
            berth_id="BERTH-1",
            eqp_num=2,
        )
        result = PlanningResult(
            run_id="RUN-1",
            timestamp=datetime(2026, 1, 1, 0, 0, 0),
            mode=None,
            allocation_groups=groups,
            bay_column_allocations=allocations,
            metrics={"vessels": {voyage_id: vessel}},
        )

        score = score_yard_plan(result)
        item = score["items"]["business_dispersion"]
        voyage_detail = item["details"]["voyages"][voyage_id]

        self.assertEqual(voyage_detail["assignedAreas"], 2)
        self.assertEqual(voyage_detail["targetAreas"], 2)
        self.assertEqual(voyage_detail["score"], 10.0)
        self.assertEqual(voyage_detail["areaMatchRaw"], 10.0)
        self.assertEqual(voyage_detail["simSafetyRaw"], 10.0)

    def test_business_dispersion_penalizes_simultaneous_loading_gap_below_four_bays(self):
        voyage_id = "V001"
        first = make_group("GA", voyage_id, 10)
        second = make_group("GB", voyage_id, 10)
        add_simultaneous_loading_conflict(first, "GB", "PAIR-1", 4)
        add_simultaneous_loading_conflict(second, "GA", "PAIR-1", 4)
        allocations = [
            make_allocation("GA", "A", 1),
            make_allocation("GB", "A", 3),
        ]
        vessel = Vessel(
            vessel_id="VESSEL-1",
            vessel_name="Test Vessel",
            voyage_id=voyage_id,
            eta=datetime(2026, 1, 1, 8, 0, 0),
            etd=datetime(2026, 1, 2, 8, 0, 0),
            berth_id="BERTH-1",
            eqp_num=1,
        )
        result = PlanningResult(
            run_id="RUN-1",
            timestamp=datetime(2026, 1, 1, 0, 0, 0),
            mode=None,
            allocation_groups=[first, second],
            bay_column_allocations=allocations,
            metrics={"vessels": {voyage_id: vessel}},
        )

        score = score_yard_plan(result)
        item = score["items"]["business_dispersion"]
        voyage_detail = item["details"]["voyages"][voyage_id]

        self.assertEqual(voyage_detail["assignedAreas"], 1)
        self.assertEqual(voyage_detail["targetAreas"], 1)
        self.assertEqual(voyage_detail["areaMatchRaw"], 10.0)
        self.assertEqual(voyage_detail["simSafetyRaw"], 4.0)
        self.assertEqual(voyage_detail["violatingPairCount"], 1)
        self.assertEqual(voyage_detail["worstGap"], 2)
        self.assertEqual(voyage_detail["score"], 7.6)
        self.assertEqual(voyage_detail["pairs"][0]["minGap"], 2)
        self.assertEqual(voyage_detail["pairs"][0]["score"], 4.0)

    def test_bay_quality_includes_rehandle_risk_for_mixed_weight_column(self):
        voyage_id = "V001"
        heavy = make_group("GH", voyage_id, 1, weight_class=WeightClass.HEAVY)
        light = make_group("GL", voyage_id, 1, weight_class=WeightClass.LIGHT)
        allocations = [
            BayColumnAllocation(
                allocation_id="ALLOC-GH",
                group_id="GH",
                yard_area_id="A",
                business_type=BusinessType.EXPORT,
                size=ContainerSize.SIZE_20,
                bay_column_details=[(1, 1)],
                notes="greedy stage2 status=greedy; placed=1/1; columns=1; exact 1: stack 1 tiers 2",
            ),
            BayColumnAllocation(
                allocation_id="ALLOC-GL",
                group_id="GL",
                yard_area_id="A",
                business_type=BusinessType.EXPORT,
                size=ContainerSize.SIZE_20,
                bay_column_details=[(1, 1)],
                notes="greedy stage2 status=greedy; placed=1/1; columns=1; exact 1: stack 1 tiers 1",
            ),
        ]
        result = PlanningResult(
            run_id="RUN-2",
            timestamp=datetime(2026, 1, 1, 0, 0, 0),
            mode=None,
            allocation_groups=[heavy, light],
            bay_column_allocations=allocations,
        )

        score = score_yard_plan(result)
        item = score["items"]["bay_quality"]

        self.assertAlmostEqual(item["details"]["rehandleRiskScore"], 0.5, places=4)
        self.assertAlmostEqual(item["details"]["rehandleRiskRaw"], 5.0, places=2)
        self.assertIn("A:1:1", item["details"]["columns"])
        self.assertEqual(item["details"]["columns"]["A:1:1"]["heavyCount"], 1)
        self.assertEqual(item["details"]["columns"]["A:1:1"]["lightCount"], 1)
        self.assertLess(item["score"], 10.0)


if __name__ == "__main__":
    unittest.main()
