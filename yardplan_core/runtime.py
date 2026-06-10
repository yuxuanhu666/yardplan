from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from yardplan_core.integrations import TOSLoader
from yardplan_core.models import (
    AllocationGroup,
    BusinessType,
    Container,
    PlannerMode,
    PlanningResult,
    logger,
)
from yardplan_core.planner import YardPlanner
from yardplan_core.visualization import YardVisualizationConfig, save_yard_visualization


def _normalize_plan_window(
    plan_start_time: Optional[datetime],
    plan_end_time: Optional[datetime],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    if plan_start_time is None and plan_end_time is None:
        return None, None
    if plan_start_time is None or plan_end_time is None:
        raise ValueError("plan_start_time 与 plan_end_time 须同时指定或同时省略")
    if plan_end_time <= plan_start_time:
        raise ValueError(
            f"plan_end_time ({plan_end_time}) 必须晚于 plan_start_time ({plan_start_time})"
        )
    if plan_start_time.tzinfo is not None:
        plan_start_time = plan_start_time.astimezone(timezone.utc).replace(tzinfo=None)
    if plan_end_time.tzinfo is not None:
        plan_end_time = plan_end_time.astimezone(timezone.utc).replace(tzinfo=None)
    return plan_start_time, plan_end_time


def _range_plan_data(result: PlanningResult) -> List[Dict[str, Any]]:
    return result.metrics.get("range_plan", {}).get("data", [])


def _normalize_key(
    loader: TOSLoader,
    raw_key: Optional[int],
    label: str,
) -> Optional[int]:
    if raw_key is None:
        return None

    key = loader._coerce_line_key(raw_key)
    if key is None:
        raise ValueError(f"Invalid {label}: {raw_key!r}")
    return key


def _apply_departed_loading_cleanup(
    *,
    loader: TOSLoader,
    yard: Any,
    vessels: Dict[str, Any],
    vessel_key: Optional[int],
    plan_start_time: Optional[datetime],
    plan_end_time: Optional[datetime],
) -> Dict[str, Any]:
    if vessel_key is None or len(vessels) != 1:
        return {
            "enabled": False,
            "reason": "cleanup is only enabled for single vessel_key planning",
        }

    target_vessel = next(iter(vessels.values()))
    plan_made_at = target_vessel.eta - timedelta(days=4)
    cleanup_start = plan_made_at
    if plan_start_time is not None and plan_start_time > cleanup_start:
        cleanup_start = plan_start_time
    cleanup_data = loader.load_departed_loading_container_ids(
        target_vessel_key=vessel_key,
        target_eta=target_vessel.eta,
        plan_start_time=cleanup_start,
        plan_end_time=target_vessel.eta,
    )

    container_ids = cleanup_data.get("container_ids", [])
    removed = 0
    missing = 0
    for container_id in container_ids:
        if yard.remove_container_by_id(container_id):
            removed += 1
        else:
            missing += 1

    candidate_vessels = cleanup_data.get("candidate_vessels", [])
    active_vessels = [
        vessel
        for vessel in candidate_vessels
        if vessel.get("loadingContainerCount", 0) > 0
    ]
    preview = ", ".join(
        f"{vessel.get('vesselVisitId')}({vessel.get('loadingContainerCount', 0)})"
        for vessel in active_vessels[:8]
    )
    print("\n  预测清场(仅装船箱):")
    print(
        f"    目标船 ETA: {target_vessel.eta}  |  清场窗口: {cleanup_start} -> {target_vessel.eta}"
    )
    print(
        f"    候选离港船: {len(candidate_vessels)}  |  有装船WQ: {len(active_vessels)}"
    )
    print(
        f"    WQ装船箱: {len(container_ids)}  |  已从在场箱移除: {removed}  |  未在堆场找到: {missing}"
    )
    if preview:
        print(f"    涉及船舶(前8): {preview}")

    return {
        "enabled": True,
        "targetVesselKey": vessel_key,
        "targetEta": target_vessel.eta.isoformat(),
        "planMadeAt": plan_made_at.isoformat(),
        "cleanupStart": cleanup_start.isoformat(),
        "cleanupEnd": target_vessel.eta.isoformat(),
        "candidateVesselCount": len(candidate_vessels),
        "activeLoadingVesselCount": len(active_vessels),
        "containerCount": len(container_ids),
        "removedCount": removed,
        "missingCount": missing,
        "activeVesselPreview": [
            {
                "vesselVisitKey": vessel.get("vesselVisitKey"),
                "vesselVisitId": vessel.get("vesselVisitId"),
                "etd": vessel.get("etd").isoformat()
                if hasattr(vessel.get("etd"), "isoformat")
                else vessel.get("etd"),
                "loadingContainerCount": vessel.get("loadingContainerCount", 0),
            }
            for vessel in active_vessels[:20]
        ],
    }


def _build_yard_busy_profile(
    *,
    loader: TOSLoader,
    yard: Any,
    vessels: Dict[str, Any],
    vessel_key: Optional[int],
    moves_per_hour: float = 20.0,
) -> Dict[str, Any]:
    if vessel_key is None or len(vessels) != 1:
        return {
            "enabled": False,
            "reason": "busy profile is only enabled for single vessel_key planning",
        }

    target_vessel = next(iter(vessels.values()))
    window_start = target_vessel.eta - timedelta(days=4)
    window_end = target_vessel.eta
    bucket_count = 4
    bucket_hours = 24.0
    buckets = [
        {
            "index": index,
            "start": window_start + timedelta(days=index),
            "end": window_start + timedelta(days=index + 1),
            "isPeak": index in {1, 2},
        }
        for index in range(bucket_count)
    ]

    wq_data = loader.load_loading_wq_containers_for_window(
        target_vessel_key=vessel_key,
        window_start=window_start,
        window_end=window_end,
    )

    area_counts: Dict[str, List[int]] = defaultdict(lambda: [0] * bucket_count)
    area_busy_hours: Dict[str, List[float]] = defaultdict(lambda: [0.0] * bucket_count)
    missing_container_count = 0
    vessels_with_loading = []

    containers_by_vessel = wq_data.get("containers_by_vessel", {})
    candidate_by_key = {
        vessel.get("vesselVisitKey"): vessel
        for vessel in wq_data.get("candidate_vessels", [])
    }

    for raw_visit_key, container_ids in containers_by_vessel.items():
        visit_key = loader._coerce_line_key(raw_visit_key)
        vessel = candidate_by_key.get(visit_key, {})
        if not container_ids:
            continue

        block_counts: Dict[str, int] = defaultdict(int)
        for container_id in container_ids:
            block_id = yard.get_container_block_id(container_id)
            if not block_id:
                missing_container_count += 1
                continue
            block_counts[block_id] += 1

        if not block_counts:
            continue

        visit_start = vessel.get("eta") or window_start
        if visit_start < window_start:
            visit_start = window_start
        if visit_start >= window_end:
            continue

        for block_id, count in block_counts.items():
            duration_hours = count / max(1e-6, moves_per_hour)
            busy_start = visit_start
            busy_end = busy_start + timedelta(hours=duration_hours)
            if busy_end <= busy_start:
                continue
            for bucket in buckets:
                overlap_start = max(busy_start, bucket["start"])
                overlap_end = min(busy_end, bucket["end"])
                if overlap_end <= overlap_start:
                    continue
                overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600.0
                index = bucket["index"]
                area_counts[block_id][index] += count
                area_busy_hours[block_id][index] += overlap_hours

        vessels_with_loading.append(
            {
                "vesselVisitKey": visit_key,
                "vesselVisitId": vessel.get("vesselVisitId"),
                "eta": vessel.get("eta").isoformat()
                if hasattr(vessel.get("eta"), "isoformat")
                else vessel.get("eta"),
                "etd": vessel.get("etd").isoformat()
                if hasattr(vessel.get("etd"), "isoformat")
                else vessel.get("etd"),
                "containerCount": len(container_ids),
                "yardContainerCount": sum(block_counts.values()),
                "blocks": dict(sorted(block_counts.items())),
            }
        )

    areas: Dict[str, Dict[str, Any]] = {}
    for block_id in sorted(area_busy_hours):
        busy_hours = area_busy_hours[block_id]
        ratios = [min(1.0, hours / bucket_hours) for hours in busy_hours]
        peak_ratio = max((ratios[index] for index in (1, 2)), default=0.0)
        capacity_factor = max(0.0, 1.0 - peak_ratio)
        hard_blocked = peak_ratio >= 0.45
        areas[block_id] = {
            "bucketContainerCounts": area_counts[block_id],
            "bucketBusyHours": [round(value, 3) for value in busy_hours],
            "bucketBusyRatios": [round(value, 4) for value in ratios],
            "peakBusyRatio": round(peak_ratio, 4),
            "capacityFactor": round(capacity_factor, 4),
            "hardBlocked": hard_blocked,
        }

    top_areas = sorted(
        areas.items(),
        key=lambda item: item[1]["peakBusyRatio"],
        reverse=True,
    )[:10]

    print("\n  箱区装船忙闲预测(日桶, 20箱/小时):")
    print(f"    窗口: {window_start} -> {window_end}")
    print(
        f"    候选装船船舶: {len(wq_data.get('candidate_vessels', []))}  |  "
        f"有装船箱且在堆场定位: {len(vessels_with_loading)}  |  未定位箱: {missing_container_count}"
    )
    if top_areas:
        preview = ", ".join(
            f"{block_id}(peak={profile['peakBusyRatio']:.2f})"
            for block_id, profile in top_areas
        )
        print(f"    高峰冲突箱区Top: {preview}")

    return {
        "enabled": True,
        "targetVesselKey": vessel_key,
        "targetEta": target_vessel.eta.isoformat(),
        "windowStart": window_start.isoformat(),
        "windowEnd": window_end.isoformat(),
        "movesPerHour": moves_per_hour,
        "buckets": [
            {
                "index": bucket["index"],
                "start": bucket["start"].isoformat(),
                "end": bucket["end"].isoformat(),
                "isPeak": bucket["isPeak"],
            }
            for bucket in buckets
        ],
        "areas": areas,
        "vessels": vessels_with_loading,
        "missingContainerCount": missing_container_count,
    }


def run_plan(
    token: Optional[str] = None,
    *,
    line_keys: Optional[int] = None,
    vessel_key: Optional[int] = None,
    type: int = 1,
    apply_to_yard: bool = False,
    save_visualization: bool = False,
    visualization_path: Optional[str] = None,
    visualization_block_ids: Optional[Sequence[str]] = None,
    visualization_use_real_coordinates: bool = True,
    plan_start_time: Optional[datetime] = None,
    plan_end_time: Optional[datetime] = None,
    return_list: bool = False,
    print_score: bool = False,
) -> Union[PlanningResult, List[Dict[str, Any]]]:
    """
    规划入口：传入一个或多个航线号，完成进出口箱堆场分配规划。

    token: 第一个参数，TOS/平台访问令牌（str），默认 None；不传时由调用方（如 FastAPI）决定是否回退到服务注册 token。
    line_keys: 航线号（keyword-only），对应请求体 serviceLineKey。
    vessel_key: VesselVisit 顶层 dbkey（keyword-only），对应请求体 vesselVisitKey；与 line_keys 二选一。
    return_list: 为 True 时直接返回 API 用的分配组列表（等同 metrics["range_plan"]["data"]）；
        为 False 时返回完整 PlanningResult（本地调试、可视化用）。

    plan_start_time / plan_end_time: 手动规划范围；仅纳入与该区间有交集的航次，
    滚动时间步亦使用该区间。二者须同时指定或同时省略。
    """

    if type not in (1, 2):
        raise ValueError(f"type 只能为 1 或 2，当前为: {type!r}")

    plan_start_time, plan_end_time = _normalize_plan_window(
        plan_start_time, plan_end_time
    )

    loader = TOSLoader(token=token)
    normalized_line_key = _normalize_key(loader, line_keys, "lineKey")
    normalized_vessel_key = _normalize_key(loader, vessel_key, "VesselVisit dbkey")

    has_line_keys = normalized_line_key is not None
    has_vessel_key = normalized_vessel_key is not None
    if has_line_keys == has_vessel_key:
        raise ValueError("Provide exactly one of line_keys or vessel_key")
    if has_vessel_key and type != 1:
        raise ValueError("vessel_key mode currently supports type=1 only")

    print("=" * 70)
    if has_vessel_key:
        print(f"  visitDbkey: {normalized_vessel_key}")
    print(f"  堆场规划  type={type}  lineKey: {normalized_line_key}")
    if plan_start_time is not None and plan_end_time is not None:
        print(f"  规划范围: {plan_start_time} → {plan_end_time}")
    print("=" * 70)

    vessels = loader.load_vessels(
        line_keys=normalized_line_key,
        vessel_key=normalized_vessel_key,
        plan_start_time=plan_start_time,
        plan_end_time=plan_end_time,
    )
    if not vessels:
        logger.warning("规划范围内无匹配船舶，规划终止")
        return PlanningResult(
            run_id="PLAN-EMPTY",
            timestamp=datetime.now(),
            mode=PlannerMode.FULL_PLAN,
        )

    horizon_start, horizon_end = loader.build_planning_horizon(
        vessels,
        plan_start_time=plan_start_time,
        plan_end_time=plan_end_time,
    )

    containers: List[Container] = []
    external_groups: List[AllocationGroup] = []

    if type == 1:
        import_containers = loader.load_discharge_containers(
            line_keys=normalized_line_key,
            vessel_key=normalized_vessel_key,
            vessels=vessels,
        )
        export_containers = loader.load_loading_containers(
            line_keys=normalized_line_key,
            vessel_key=normalized_vessel_key,
            vessels=vessels,
        )
        containers = import_containers + export_containers
        if not containers:
            logger.warning("未加载到任何进出口箱，规划终止")
            return PlanningResult(
                run_id="PLAN-EMPTY",
                timestamp=datetime.now(),
                mode=PlannerMode.FULL_PLAN,
            )
        if has_vessel_key:
            container_visit_ids = {container.voyage_id for container in containers}
            vessels = {
                visit_id: vessel
                for visit_id, vessel in vessels.items()
                if visit_id in container_visit_ids
            }
            horizon_start, horizon_end = loader.build_planning_horizon(
                vessels,
                plan_start_time=plan_start_time,
                plan_end_time=plan_end_time,
            )
    else:
        external_groups = loader.load_external_allocation_groups(
            normalized_line_key,
            vessels,
        )
        if not external_groups:
            logger.warning("未加载到任何外部分配组，规划终止")
            return PlanningResult(
                run_id="PLAN-EMPTY",
                timestamp=datetime.now(),
                mode=PlannerMode.FULL_PLAN,
            )

    from useable_space import YardSpace

    yard = YardSpace.load()
    yard_busy_profile = _build_yard_busy_profile(
        loader=loader,
        yard=yard,
        vessels=vessels,
        vessel_key=normalized_vessel_key,
    )
    cleanup_metrics = _apply_departed_loading_cleanup(
        loader=loader,
        yard=yard,
        vessels=vessels,
        vessel_key=normalized_vessel_key,
        plan_start_time=plan_start_time,
        plan_end_time=plan_end_time,
    )
    yard.print_summary()

    block_ids = sorted({key[0] for key in yard.stacks.keys()})

    def _serial(block_id: str) -> int:
        digits = "".join(char for char in block_id if char.isdigit())
        return int(digits) if digits else 0

    block_business_types = {
        block_id: (
            BusinessType.IMPORT if _serial(block_id) % 2 == 1 else BusinessType.EXPORT
        )
        for block_id in block_ids
    }

    planner = YardPlanner()
    if type == 1:
        result = planner.plan_with_yard_space(
            yard=yard,
            containers=containers,
            block_business_types=block_business_types,
            vessels=vessels,
            yard_busy_profile=yard_busy_profile,
            mode=PlannerMode.FULL_PLAN,
            apply_to_yard=apply_to_yard,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            print_score=print_score,
        )
    else:
        result = planner.plan_groups_with_yard_space(
            yard=yard,
            groups=external_groups,
            block_business_types=block_business_types,
            vessels=vessels,
            mode=PlannerMode.FULL_PLAN,
            apply_to_yard=apply_to_yard,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            print_score=print_score,
        )

    result.metrics["yard_cleanup"] = cleanup_metrics
    result.metrics["yard_busy_profile"] = yard_busy_profile

    range_items = _range_plan_data(result)
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

    if save_visualization:
        output_path = visualization_path or _default_visualization_path(result.run_id)
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        config = YardVisualizationConfig(
            prefer_real_coordinates=visualization_use_real_coordinates
        )
        try:
            save_yard_visualization(
                yard=yard,
                result=result,
                save_path=output_path,
                block_ids=list(visualization_block_ids)
                if visualization_block_ids is not None
                else None,
                config=config,
            )
            result.metrics["visualization_path"] = output_path
            print(f"\n  可视化结果已保存: {output_path}")
        except Exception as exc:
            logger.warning("生成可视化失败: %s", exc)
            print(f"\n  可视化生成失败: {exc}")

    if return_list:
        return range_items
    return result


def _default_visualization_path(run_id: str) -> str:
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
    )
    os.makedirs(output_dir, exist_ok=True)
    safe_run_id = run_id or datetime.now().strftime("PLAN-%Y%m%d-%H%M%S")
    return os.path.join(output_dir, f"{safe_run_id}.png")


__all__ = ["run_plan"]
