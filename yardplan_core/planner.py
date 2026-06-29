from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from yardplan_core.allocation import AllocationEngine, RollingWindowPlanner
from yardplan_core.grouping import GroupingEngine
from yardplan_core.integrations import YardSpaceAdapter
from yardplan_core.models import (
    AllocationGroup,
    BusinessType,
    MAX_TIERS_PER_COLUMN,
    PlannerMode,
    PlanningResult,
    WeightClass,
    YardArea,
    Container,
    Vessel,
    logger,
)
from yardplan_score import format_score_report, score_yard_plan


class ResultFormatter:
    """Pretty printer for planning results."""

    @staticmethod
    def build_range_plan(result: PlanningResult) -> Dict[str, List[Dict[str, Any]]]:
        group_by_id: Dict[str, AllocationGroup] = {
            group.group_id: group for group in result.allocation_groups
        }
        data: List[Dict[str, Any]] = []

        for idx, allocation in enumerate(result.bay_column_allocations, start=1):
            group = group_by_id.get(allocation.group_id)
            if group is None:
                continue

            if allocation.bay_stack_details:
                range_list = [
                    ResultFormatter._build_range_item(
                        allocation.yard_area_id,
                        bay_spec,
                        end_stack - start_stack + 1,
                        stack_start=start_stack,
                        stack_end=end_stack,
                    )
                    for bay_spec, start_stack, end_stack in allocation.bay_stack_details
                ]
            else:
                range_list = [
                    ResultFormatter._build_range_item(
                        allocation.yard_area_id,
                        bay_spec,
                        columns_used,
                    )
                    for bay_spec, columns_used in allocation.bay_column_details
                ]
            if not range_list:
                continue

            data.append(
                {
                    "groupKey": idx,
                    "groupId": idx,
                    "rangeList": range_list,
                    "filter": ResultFormatter._build_filter(group),
                }
            )

        return {"data": data}

    @staticmethod
    def _build_range_item(
        block_id: str,
        bay_spec: Any,
        columns_used: int,
        stack_start: Optional[int] = None,
        stack_end: Optional[int] = None,
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
            "startStackIndex": int(stack_start) if stack_start is not None else 1,
            "endStackIndex": int(stack_end) if stack_end is not None else max(1, int(columns_used)),
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
            values: List[Any] = []
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
            ResultFormatter._normalise_weight(container.raw_weight)
            for container in containers
            if container.raw_weight is not None
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
            "bReefer": any(container.is_reefer for container in containers),
            "bHazardous": any(container.is_hazardous for container in containers),
            "bDamage": any(container.is_damage for container in containers),
            "bHigh": any(container.is_high for container in containers),
            "bGauge": any(container.is_gauge for container in containers),
            "ownerCompany": unique("owner_company"),
            "lineCompany": unique("line_company"),
            "truckCompany": unique("truck_company"),
            "belongerCompany": unique("belonger_company"),
            "bDirty": any(container.is_dirty for container in containers),
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
    def print_summary(result: PlanningResult) -> None:
        print("\n" + "=" * 80)
        print("YARD SPACE ALLOCATION PLANNING RESULT")
        print("=" * 80)
        print(f"Run ID: {result.run_id} | Mode: {result.mode.value} | Time: {result.timestamp}")
        print(f"Groups generated: {len(result.allocation_groups)}")
        print(f"Area assignments: {len(result.area_assignments)}")
        print(f"Bay/column allocations: {len(result.bay_column_allocations)}")
        print(f"Unassigned groups: {len(result.unassigned_groups)}")

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"   - {warning}")

        print("\n" + "=" * 80)

    @staticmethod
    def print_yard_busy_profile(
        yard_busy_profile: Optional[Dict[str, Any]],
        yard_areas: List[YardArea],
    ) -> None:
        if not yard_busy_profile or not yard_busy_profile.get("enabled"):
            return

        buckets = yard_busy_profile.get("buckets") or []
        areas = yard_busy_profile.get("areas") or {}
        bucket_headers = []
        for bucket in buckets:
            start = str(bucket.get("start") or "")[:10]
            end = str(bucket.get("end") or "")[:10]
            bucket_headers.append(f"{start}~{end}")

        print("\n" + "=" * 108)
        print("【WQ估算】规划前4天 · 按天桶的箱区忙闲程度")
        print("=" * 108)
        print(
            f"窗口: {yard_busy_profile.get('windowStart')} ~ {yard_busy_profile.get('windowEnd')}"
            f"  |  moves_per_hour={yard_busy_profile.get('movesPerHour')}"
        )
        print(
            f"未定位WQ箱: {yard_busy_profile.get('missingContainerCount', 0)}"
            f"  |  有WQ且可定位船舶: {len(yard_busy_profile.get('vessels') or [])}"
        )
        if bucket_headers:
            print("日桶: " + " | ".join(f"D{idx}:{header}" for idx, header in enumerate(bucket_headers)))
        hard_block_indices = yard_busy_profile.get("hardBlockBucketIndices") or [1, 2, 3]
        hard_block_threshold = float(yard_busy_profile.get("hardBlockThreshold") or 0.3)
        print(
            "禁用规则: "
            + "/".join(f"D{index}" for index in hard_block_indices)
            + f" 任一忙闲比例 > {hard_block_threshold:.2f} 则 hardBlocked=True"
        )
        core_start = yard_busy_profile.get("coreReceivingStart")
        core_end = yard_busy_profile.get("coreReceivingEnd")
        if core_start and core_end:
            vessel_overlap_threshold = float(
                yard_busy_profile.get("vesselOverlapHardBlockThreshold") or 0.2
            )
            print(
                f"船期重叠禁用: 其他船 ETA~ETD 与 {core_start} ~ {core_end} "
                f"重叠且箱区忙闲比例 > {vessel_overlap_threshold:.2f} 时 hardBlocked=True"
            )

        area_rows: List[Dict[str, Any]] = []
        for area in sorted(yard_areas, key=lambda item: item.area_id):
            profile = areas.get(area.area_id) or {}
            counts = list(profile.get("bucketContainerCounts") or [])
            hours = list(profile.get("bucketBusyHours") or [])
            ratios = list(profile.get("bucketBusyRatios") or [])
            while len(counts) < 4:
                counts.append(0)
            while len(hours) < 4:
                hours.append(0.0)
            while len(ratios) < 4:
                ratios.append(0.0)
            area_rows.append(
                {
                    "area": area,
                    "counts": counts,
                    "hours": hours,
                    "ratios": ratios,
                    "peak": float(profile.get("peakBusyRatio") or 0.0),
                    "capacity_factor": float(profile.get("capacityFactor") or 1.0),
                    "hard_blocked": bool(profile.get("hardBlocked")),
                    "busy_ratio_blocked": bool(profile.get("busyRatioBlocked")),
                    "vessel_overlap_blocked": bool(
                        profile.get("vesselOverlapBlocked")
                    ),
                    "vessel_overlap_blocks": list(
                        profile.get("vesselOverlapBlocks") or []
                    ),
                }
            )

        busy_rows = [
            row for row in area_rows if row["peak"] > 0.0 or row["hard_blocked"]
        ]
        if busy_rows:
            print("\n阻塞峰值忙闲摘要(按peak降序):")
            for row in sorted(
                busy_rows,
                key=lambda item: (-item["peak"], item["area"].area_id),
            ):
                print(
                    f"{row['area'].area_id} "
                    f"peak={row['peak']:.2f} "
                    f"hardBlocked={row['hard_blocked']} "
                    f"ratioBlocked={row['busy_ratio_blocked']} "
                    f"vesselOverlapBlocked={row['vessel_overlap_blocked']} "
                    f"capacityFactor={row['capacity_factor']:.2f}"
                )
                if row["vessel_overlap_blocks"]:
                    preview = ", ".join(
                        f"{item.get('vesselVisitId')}({item.get('containerCount')})"
                        for item in row["vessel_overlap_blocks"][:5]
                    )
                    print(f"    overlapVessels: {preview}")

        print(
            f"{'箱区':<10} {'业态':<8} "
            f"{'D0箱':>7} {'D0小时':>8} {'D0忙闲':>8} "
            f"{'D1箱':>7} {'D1小时':>8} {'D1忙闲':>8} "
            f"{'D2箱':>7} {'D2小时':>8} {'D2忙闲':>8} "
            f"{'D3箱':>7} {'D3小时':>8} {'D3忙闲':>8} "
            f"{'峰值':>8} {'硬阻塞':>8} {'原因':>14}"
        )
        print("-" * 148)

        total_counts = [0, 0, 0, 0]
        total_busy_hours = [0.0, 0.0, 0.0, 0.0]
        for row in area_rows:
            area = row["area"]
            counts = row["counts"]
            hours = row["hours"]
            ratios = row["ratios"]
            for index in range(4):
                total_counts[index] += float(counts[index] or 0.0)
                total_busy_hours[index] += float(hours[index] or 0.0)

            business = "进口" if area.business_type == BusinessType.IMPORT else "出口"
            reason = []
            if row["busy_ratio_blocked"]:
                reason.append("ratio")
            if row["vessel_overlap_blocked"]:
                reason.append("vessel")
            print(
                f"{area.area_id:<10} {business:<8} "
                f"{float(counts[0]):7.1f} {float(hours[0]):8.2f} {float(ratios[0]):8.3f} "
                f"{float(counts[1]):7.1f} {float(hours[1]):8.2f} {float(ratios[1]):8.3f} "
                f"{float(counts[2]):7.1f} {float(hours[2]):8.2f} {float(ratios[2]):8.3f} "
                f"{float(counts[3]):7.1f} {float(hours[3]):8.2f} {float(ratios[3]):8.3f} "
                f"{row['peak']:8.3f} {str(row['hard_blocked']):>8} {','.join(reason):>14}"
            )

        bucket_hours = 24.0
        total_ratios = [hours / bucket_hours for hours in total_busy_hours]
        print("-" * 148)
        print(
            f"{'合计':<10} {'':<8} "
            f"{total_counts[0]:7.1f} {total_busy_hours[0]:8.2f} {total_ratios[0]:8.3f} "
            f"{total_counts[1]:7.1f} {total_busy_hours[1]:8.2f} {total_ratios[1]:8.3f} "
            f"{total_counts[2]:7.1f} {total_busy_hours[2]:8.2f} {total_ratios[2]:8.3f} "
            f"{total_counts[3]:7.1f} {total_busy_hours[3]:8.2f} {total_ratios[3]:8.3f} "
            f"{max(total_ratios, default=0.0):8.3f} {'':>8}"
        )

        print("\n按时间桶展开(箱区按桶内忙闲比例降序):")
        for index, bucket in enumerate(buckets[:4]):
            start = str(bucket.get("start") or "")
            end = str(bucket.get("end") or "")
            print(f"\nD{index}  {start} ~ {end}")
            print(
                f"{'箱区':<10} {'业态':<8} {'WQ箱':>7} "
                f"{'忙碌小时':>9} {'忙闲比例':>9} {'peak':>8} "
                f"{'hardBlocked':>12} {'原因':>14}"
            )
            print("-" * 74)
            for row in sorted(
                area_rows,
                key=lambda item: (
                    -float(item["ratios"][index] or 0.0),
                    -float(item["counts"][index] or 0.0),
                    item["area"].area_id,
                ),
            ):
                area = row["area"]
                business = (
                    "进口"
                    if area.business_type == BusinessType.IMPORT
                    else "出口"
                )
                reason = []
                if row["busy_ratio_blocked"]:
                    reason.append("ratio")
                if row["vessel_overlap_blocked"]:
                    reason.append("vessel")
                print(
                    f"{area.area_id:<10} {business:<8} "
                    f"{float(row['counts'][index] or 0.0):7.1f} "
                    f"{float(row['hours'][index] or 0.0):9.2f} "
                    f"{float(row['ratios'][index] or 0.0):9.3f} "
                    f"{row['peak']:8.3f} {str(row['hard_blocked']):>12} "
                    f"{','.join(reason):>14}"
                )
        print("=" * 108 + "\n")


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
        vessels: Optional[Dict[str, Vessel]] = None,
        mode: PlannerMode = PlannerMode.FULL_PLAN,
        horizon_start: Optional[datetime] = None,
        horizon_end: Optional[datetime] = None,
        yard_busy_profile: Optional[Dict[str, Any]] = None,
        print_score: bool = False,
    ) -> PlanningResult:
        run_id = f"PLAN-{uuid.uuid4().hex[:12].upper()}"
        timestamp = datetime.now()

        if mode == PlannerMode.GROUP_ONLY:
            groups = self.grouping_engine.group_containers(containers)
            result = PlanningResult(
                run_id=run_id,
                timestamp=timestamp,
                mode=mode,
                allocation_groups=groups,
            )
            result.metrics["range_plan"] = {"data": []}
            result.metrics["data"] = []
            self.formatter.print_summary(result)
            return result

        if not horizon_start or not horizon_end:
            all_times = [container.eta for container in containers if container.eta] + [
                container.etd for container in containers if container.etd
            ]
            horizon_start = min(all_times) if all_times else datetime.now()
            horizon_end = (
                max(all_times) if all_times else horizon_start + timedelta(hours=48)
            )

        time_steps = self.rolling_planner.generate_time_steps(
            horizon_start,
            horizon_end,
        )
        windows = self.rolling_planner.generate_rolling_windows(time_steps)
        self.rolling_planner.update_future_capacity(windows, yard_areas, [])

        groups = self.grouping_engine.group_containers(containers)
        self.formatter.print_yard_busy_profile(yard_busy_profile, yard_areas)
        workload_snapshot = self.allocation_engine.stage1.build_workload_snapshot(
            yard_areas=yard_areas,
            time_steps=time_steps,
            vessels=vessels or {},
            groups=groups,
            plan_start_time=horizon_start,
            plan_end_time=horizon_end,
        )
        area_assignments, bay_allocations, unassigned = self.allocation_engine.allocate(
            groups,
            yard_areas,
            workload_snapshot=workload_snapshot,
        )

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
        result.metrics["vessels"] = vessels or {}

        self.formatter.print_summary(result)
        if print_score:
            score = score_yard_plan(
                result,
                yard_areas=yard_areas,
                workload_snapshot=workload_snapshot,
                vessels=vessels,
            )
            print(format_score_report(score, indent="  "))
        return result

    def plan_groups(
        self,
        groups: List[AllocationGroup],
        yard_areas: List[YardArea],
        vessels: Optional[Dict[str, Vessel]] = None,
        mode: PlannerMode = PlannerMode.FULL_PLAN,
        horizon_start: Optional[datetime] = None,
        horizon_end: Optional[datetime] = None,
        yard_busy_profile: Optional[Dict[str, Any]] = None,
        print_score: bool = False,
    ) -> PlanningResult:
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
            horizon_end = (
                max(all_times) if all_times else horizon_start + timedelta(hours=48)
            )

        time_steps = self.rolling_planner.generate_time_steps(
            horizon_start,
            horizon_end,
        )
        windows = self.rolling_planner.generate_rolling_windows(time_steps)
        self.rolling_planner.update_future_capacity(windows, yard_areas, [])

        self.formatter.print_yard_busy_profile(yard_busy_profile, yard_areas)
        workload_snapshot = self.allocation_engine.stage1.build_workload_snapshot(
            yard_areas=yard_areas,
            time_steps=time_steps,
            vessels=vessels or {},
            groups=groups,
            plan_start_time=horizon_start,
            plan_end_time=horizon_end,
        )
        area_assignments, bay_allocations, unassigned = self.allocation_engine.allocate(
            groups,
            yard_areas,
            workload_snapshot=workload_snapshot,
        )
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
        result.metrics["vessels"] = vessels or {}

        self.formatter.print_summary(result)
        if print_score:
            score = score_yard_plan(
                result,
                yard_areas=yard_areas,
                workload_snapshot=workload_snapshot,
                vessels=vessels,
            )
            print(format_score_report(score, indent="  "))
        return result

    def plan_groups_with_yard_space(
        self,
        yard: Any,
        groups: List[AllocationGroup],
        block_business_types: Dict[str, BusinessType],
        vessels: Optional[Dict[str, Vessel]] = None,
        block_ids: Optional[List[str]] = None,
        yard_busy_profile: Optional[Dict[str, Any]] = None,
        mode: PlannerMode = PlannerMode.FULL_PLAN,
        apply_to_yard: bool = False,
        horizon_start: Optional[datetime] = None,
        horizon_end: Optional[datetime] = None,
        print_score: bool = False,
    ) -> PlanningResult:
        yard_areas, _slot_registry = YardSpaceAdapter.build_yard_areas(
            yard,
            block_business_types,
            block_ids,
        )
        self._attach_yard_busy_profile(yard_areas, yard_busy_profile)
        result = self.plan_groups(
            groups=groups,
            yard_areas=yard_areas,
            vessels=vessels,
            mode=mode,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            yard_busy_profile=yard_busy_profile,
            print_score=print_score,
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
        vessels: Optional[Dict[str, Vessel]] = None,
        block_ids: Optional[List[str]] = None,
        yard_busy_profile: Optional[Dict[str, Any]] = None,
        mode: PlannerMode = PlannerMode.FULL_PLAN,
        apply_to_yard: bool = False,
        horizon_start: Optional[datetime] = None,
        horizon_end: Optional[datetime] = None,
        print_score: bool = False,
    ) -> PlanningResult:
        yard_areas, _slot_registry = YardSpaceAdapter.build_yard_areas(
            yard,
            block_business_types,
            block_ids,
        )
        self._attach_yard_busy_profile(yard_areas, yard_busy_profile)
        result = self.plan(
            containers=containers,
            yard_areas=yard_areas,
            vessels=vessels,
            mode=mode,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            yard_busy_profile=yard_busy_profile,
            print_score=print_score,
        )
        if apply_to_yard:
            assignments = YardSpaceAdapter.apply_allocation(result, yard)
            result.metrics["slot_assignments"] = assignments
            logger.info(
                f"已将规划结果写回 YardSpace: {len(assignments)} 个容器获得槽位"
            )
        return result

    @staticmethod
    def _attach_yard_busy_profile(
        yard_areas: List[YardArea],
        yard_busy_profile: Optional[Dict[str, Any]],
    ) -> None:
        if not yard_busy_profile or not yard_busy_profile.get("enabled"):
            return
        area_profiles = yard_busy_profile.get("areas") or {}
        for area in yard_areas:
            profile = area_profiles.get(area.area_id)
            if not profile:
                continue
            area._stage1_busy_profile = profile
            area._stage1_peak_busy_ratio = float(
                profile.get("planningPeakBusyRatio", profile.get("peakBusyRatio"))
                or 0.0
            )
            area._stage1_capacity_factor = float(profile.get("capacityFactor") or 1.0)
            area._stage1_hard_blocked = bool(profile.get("hardBlocked"))


__all__ = ["ResultFormatter", "YardPlanner"]
