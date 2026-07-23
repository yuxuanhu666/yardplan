from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from yardplan_core.integrations import TOSLoader
from yardplan_core.models import (
    BusinessType,
    Container,
    ContainerSize,
    ContainerType,
    PlannerMode,
    PlanningResult,
    Vessel,
    WeightClass,
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


@dataclass(frozen=True)
class _PodDemand:
    pod: str
    count_20ft: int
    count_40ft: int


@dataclass(frozen=True)
class _DemandConfig:
    import_block_ids: Tuple[str, ...]
    import_count_20ft: int
    import_count_40ft: int
    export_block_ids: Tuple[str, ...]
    export_pod_demands: Tuple[_PodDemand, ...]


def _normalize_single_key(
    loader: TOSLoader,
    raw_key: Any,
    label: str,
) -> Optional[int]:
    if raw_key is None:
        return None

    keys = loader._coerce_line_key_set(raw_key)
    if not keys:
        raise ValueError(f"Invalid {label}: {raw_key!r}")
    if len(keys) != 1:
        raise ValueError(f"Exactly one {label} is currently supported: {raw_key!r}")
    return next(iter(keys))


def _resolve_plan_type(
    *,
    legacy_type: Optional[int],
    is_auto_group: Optional[int],
    is_auto_range: Optional[int],
) -> int:
    if is_auto_group is None and is_auto_range is None:
        resolved = 1 if legacy_type is None else legacy_type
    else:
        group_flag = _normalize_binary_flag(is_auto_group, "isAutoGroup")
        range_flag = _normalize_binary_flag(is_auto_range, "isAutoRange")
        if group_flag + range_flag != 1:
            raise ValueError("Exactly one of isAutoGroup and isAutoRange must be 1")
        resolved = 1 if group_flag == 1 else 2
        if legacy_type is not None and legacy_type != resolved:
            raise ValueError(
                f"type={legacy_type} conflicts with isAutoGroup/isAutoRange"
            )

    if resolved not in (1, 2):
        raise ValueError(f"type must be 1 or 2, got {resolved!r}")
    return resolved


def _normalize_binary_flag(raw_value: Optional[int], label: str) -> int:
    if raw_value is None:
        return 0
    if isinstance(raw_value, bool):
        return int(raw_value)
    if raw_value in (0, 1):
        return int(raw_value)
    raise ValueError(f"{label} must be 0 or 1, got {raw_value!r}")


def _normalize_nonnegative_count(raw_value: Any, label: str) -> int:
    if isinstance(raw_value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if value < 0 or value != raw_value:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _normalize_block_ids(raw_value: Any, label: str) -> Tuple[str, ...]:
    if raw_value is None:
        return ()
    if not isinstance(raw_value, (list, tuple, set)):
        raise ValueError(f"{label} must be an array")
    result: List[str] = []
    for item in raw_value:
        block_id = str(item or "").strip()
        if not block_id:
            raise ValueError(f"{label} cannot contain an empty blockId")
        if block_id not in result:
            result.append(block_id)
    return tuple(result)


def _normalize_demand_config(
    import_info: Optional[Mapping[str, Any]],
    export_info: Optional[Mapping[str, Any]],
) -> Optional[_DemandConfig]:
    if import_info is None and export_info is None:
        return None
    if import_info is None or export_info is None:
        raise ValueError("importInfo and exportInfo must be provided together")
    if not isinstance(import_info, Mapping) or not isinstance(export_info, Mapping):
        raise ValueError("importInfo and exportInfo must be objects")

    import_blocks = _normalize_block_ids(
        import_info.get("blockIdList"), "importInfo.blockIdList"
    )
    export_blocks = _normalize_block_ids(
        export_info.get("blockIdList"), "exportInfo.blockIdList"
    )
    if not export_blocks:
        raise ValueError("exportInfo.blockIdList must contain at least one blockId")

    raw_pod_list = export_info.get("podList")
    if not isinstance(raw_pod_list, list) or not raw_pod_list:
        raise ValueError("exportInfo.podList must contain at least one item")

    pod_totals: Dict[str, List[int]] = {}
    for index, item in enumerate(raw_pod_list):
        if not isinstance(item, Mapping):
            raise ValueError(f"exportInfo.podList[{index}] must be an object")
        pod = str(item.get("pod") or "").strip()
        if not pod:
            raise ValueError(f"exportInfo.podList[{index}].pod is required")
        counts = pod_totals.setdefault(pod, [0, 0])
        counts[0] += _normalize_nonnegative_count(
            item.get("cnt20Ft"), f"exportInfo.podList[{index}].cnt20Ft"
        )
        counts[1] += _normalize_nonnegative_count(
            item.get("cnt40Ft"), f"exportInfo.podList[{index}].cnt40Ft"
        )

    return _DemandConfig(
        import_block_ids=import_blocks,
        import_count_20ft=_normalize_nonnegative_count(
            import_info.get("cnt20Ft"), "importInfo.cnt20Ft"
        ),
        import_count_40ft=_normalize_nonnegative_count(
            import_info.get("cnt40Ft"), "importInfo.cnt40Ft"
        ),
        export_block_ids=export_blocks,
        export_pod_demands=tuple(
            _PodDemand(pod=pod, count_20ft=counts[0], count_40ft=counts[1])
            for pod, counts in pod_totals.items()
        ),
    )


def _is_ordinary_container(container: Container) -> bool:
    return (
        container.container_type == ContainerType.DRY
        and not container.is_reefer
        and not container.is_hazardous
        and not container.is_damage
        and not container.is_high
        and not container.is_gauge
        and not container.is_dirty
    )


def _ordinary_group_key(container: Container) -> Tuple[Any, ...]:
    common = (
        container.line_key,
        container.size,
        container.container_type,
        container.weight_class,
    )
    if container.business_type == BusinessType.EXPORT:
        return (*common, container.pod)
    return (*common, container.consignee)


def _make_generic_virtual_container(
    *,
    index: int,
    business_type: BusinessType,
    size: ContainerSize,
    pod: Optional[str],
    vessels: Dict[str, Vessel],
    line_key: Optional[int],
    fallback_containers: Sequence[Container],
) -> Container:
    source = next(
        (
            container
            for container in fallback_containers
            if container.business_type == business_type
            and _is_ordinary_container(container)
        ),
        None,
    )
    if source is not None:
        return replace(
            source,
            container_id=f"VIRTUAL-{business_type.value}-{size.value}-{index:06d}",
            size=size,
            container_type=ContainerType.DRY,
            weight_class=source.weight_class,
            destination_port=pod if business_type == BusinessType.EXPORT else None,
            pod=pod if business_type == BusinessType.EXPORT else None,
            iso_type="20GP" if size == ContainerSize.SIZE_20 else "40GP",
            current_block=None,
            current_bay=None,
            current_row=None,
            current_tier=None,
            is_reefer=False,
            is_hazardous=False,
            is_damage=False,
            is_high=False,
            is_gauge=False,
        )

    vessel = next(iter(vessels.values()))
    return Container(
        container_id=f"VIRTUAL-{business_type.value}-{size.value}-{index:06d}",
        size=size,
        container_type=ContainerType.DRY,
        weight_class=WeightClass.LIGHT,
        business_type=business_type,
        voyage_id=vessel.voyage_id,
        line_key=line_key,
        vessel_id=vessel.vessel_id,
        eta=vessel.eta,
        etd=vessel.etd,
        destination_port=pod if business_type == BusinessType.EXPORT else None,
        receiving_start=vessel.eta if business_type == BusinessType.EXPORT else None,
        iso_type="20GP" if size == ContainerSize.SIZE_20 else "40GP",
        pod=pod if business_type == BusinessType.EXPORT else None,
        raw_weight=10000.0,
    )


def _augment_containers_to_minimum_demands(
    *,
    containers: Sequence[Container],
    vessels: Dict[str, Vessel],
    line_key: Optional[int],
    demand: Optional[_DemandConfig],
    random_seed: str,
) -> Tuple[List[Container], Dict[str, Any]]:
    result = list(containers)
    if demand is None:
        return result, {"enabled": False, "virtualContainerCount": 0, "items": []}

    rng = random.Random(random_seed)
    summaries: List[Dict[str, Any]] = []
    virtual_index = 0

    requested_items = [
        (BusinessType.IMPORT, ContainerSize.SIZE_20, None, demand.import_count_20ft),
        (BusinessType.IMPORT, ContainerSize.SIZE_40, None, demand.import_count_40ft),
    ]
    for pod_demand in demand.export_pod_demands:
        requested_items.extend(
            [
                (
                    BusinessType.EXPORT,
                    ContainerSize.SIZE_20,
                    pod_demand.pod,
                    pod_demand.count_20ft,
                ),
                (
                    BusinessType.EXPORT,
                    ContainerSize.SIZE_40,
                    pod_demand.pod,
                    pod_demand.count_40ft,
                ),
            ]
        )

    actual_containers = list(containers)
    for business_type, size, pod, requested_count in requested_items:
        matching_actual = [
            container
            for container in actual_containers
            if container.business_type == business_type
            and container.size == size
            and (
                business_type == BusinessType.IMPORT
                or str(container.pod or "").strip() == pod
            )
        ]
        actual_count = len(matching_actual)
        extra_count = max(0, requested_count - actual_count)

        representatives: Dict[Tuple[Any, ...], Container] = {}
        for container in matching_actual:
            if _is_ordinary_container(container):
                representatives.setdefault(_ordinary_group_key(container), container)
        candidate_groups = list(representatives.values())

        for _ in range(extra_count):
            virtual_index += 1
            if candidate_groups:
                source = rng.choice(candidate_groups)
                virtual = replace(
                    source,
                    container_id=(
                        f"VIRTUAL-{business_type.value}-{size.value}-{virtual_index:06d}"
                    ),
                    current_block=None,
                    current_bay=None,
                    current_row=None,
                    current_tier=None,
                )
            else:
                virtual = _make_generic_virtual_container(
                    index=virtual_index,
                    business_type=business_type,
                    size=size,
                    pod=pod,
                    vessels=vessels,
                    line_key=line_key,
                    fallback_containers=actual_containers,
                )
            result.append(virtual)

        summaries.append(
            {
                "businessType": business_type.value,
                "sizeFt": size.value,
                "pod": pod,
                "requestedCount": requested_count,
                "actualCount": actual_count,
                "plannedCount": max(requested_count, actual_count),
                "virtualCount": extra_count,
                "candidateOrdinaryGroupCount": len(candidate_groups),
            }
        )

    return result, {
        "enabled": True,
        "virtualContainerCount": virtual_index,
        "items": summaries,
    }


def _serial_number(block_id: str) -> int:
    digits = "".join(char for char in block_id if char.isdigit())
    return int(digits) if digits else 0


def _resolve_block_scope(
    yard: Any,
    demand: Optional[_DemandConfig],
) -> Tuple[Dict[str, BusinessType], List[str], Dict[BusinessType, Set[str]]]:
    all_block_ids = sorted({key[0] for key in yard.stacks.keys()})
    default_types = {
        block_id: (
            BusinessType.IMPORT
            if _serial_number(block_id) % 2 == 1
            else BusinessType.EXPORT
        )
        for block_id in all_block_ids
    }
    if demand is None:
        return default_types, all_block_ids, {
            BusinessType.IMPORT: {
                block_id
                for block_id, business_type in default_types.items()
                if business_type == BusinessType.IMPORT
            },
            BusinessType.EXPORT: {
                block_id
                for block_id, business_type in default_types.items()
                if business_type == BusinessType.EXPORT
            },
        }

    canonical = {block_id.casefold(): block_id for block_id in all_block_ids}

    def resolve_requested(raw_ids: Sequence[str], label: str) -> Set[str]:
        resolved: Set[str] = set()
        missing: List[str] = []
        for raw_id in raw_ids:
            actual = canonical.get(raw_id.casefold())
            if actual is None:
                missing.append(raw_id)
            else:
                resolved.add(actual)
        if missing:
            raise ValueError(f"Unknown blockId in {label}: {sorted(missing)}")
        return resolved

    export_blocks = resolve_requested(
        demand.export_block_ids, "exportInfo.blockIdList"
    )
    import_blocks = (
        resolve_requested(demand.import_block_ids, "importInfo.blockIdList")
        if demand.import_block_ids
        else {
            block_id
            for block_id, business_type in default_types.items()
            if business_type == BusinessType.IMPORT and block_id not in export_blocks
        }
    )
    overlap = import_blocks & export_blocks
    if overlap:
        raise ValueError(
            f"Import and export block lists must be disjoint: {sorted(overlap)}"
        )

    block_business_types = {
        **{block_id: BusinessType.IMPORT for block_id in import_blocks},
        **{block_id: BusinessType.EXPORT for block_id in export_blocks},
    }
    selected_block_ids = sorted(block_business_types)
    if not selected_block_ids:
        raise ValueError("No yard blocks are available for planning")
    return block_business_types, selected_block_ids, {
        BusinessType.IMPORT: import_blocks,
        BusinessType.EXPORT: export_blocks,
    }


def _apply_departed_loading_cleanup(
    *,
    loader: TOSLoader,
    yard: Any,
    vessels: Dict[str, Any],
    vessel_key: Optional[int],
) -> Dict[str, Any]:
    if vessel_key is None or len(vessels) != 1:
        return {
            "enabled": False,
            "reason": "cleanup is only enabled for single vessel_key planning",
        }

    target_vessel = next(iter(vessels.values()))
    cleanup_start = target_vessel.eta - timedelta(days=4)
    cleanup_end = target_vessel.eta
    cleanup_data = loader.load_departed_loading_container_ids(
        target_vessel_key=vessel_key,
        target_eta=target_vessel.eta,
        plan_start_time=cleanup_start,
        plan_end_time=cleanup_end,
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
        "cleanupStart": cleanup_start.isoformat(),
        "cleanupEnd": cleanup_end.isoformat(),
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
    core_receiving_start = buckets[1]["start"]
    core_receiving_end = buckets[2]["end"]

    wq_data = loader.load_loading_wq_containers_for_window(
        target_vessel_key=vessel_key,
        window_start=window_start,
        window_end=window_end,
    )

    area_counts: Dict[str, List[float]] = defaultdict(lambda: [0.0] * bucket_count)
    area_busy_hours: Dict[str, List[float]] = defaultdict(lambda: [0.0] * bucket_count)
    area_vessel_overlap_blocks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
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

        visit_start = vessel.get("eta")
        visit_end = vessel.get("etd")
        if visit_end is None:
            continue
        overlaps_core_receiving = (
            visit_start is not None
            and visit_start < core_receiving_end
            and visit_end > core_receiving_start
        )

        for block_id, count in block_counts.items():
            if overlaps_core_receiving:
                area_vessel_overlap_blocks[block_id].append(
                    {
                        "vesselVisitKey": visit_key,
                        "vesselVisitId": vessel.get("vesselVisitId"),
                        "eta": visit_start.isoformat()
                        if hasattr(visit_start, "isoformat")
                        else visit_start,
                        "etd": visit_end.isoformat()
                        if hasattr(visit_end, "isoformat")
                        else visit_end,
                        "containerCount": count,
                    }
                )

            duration_hours = count / max(1e-6, moves_per_hour)
            busy_end = min(visit_end, window_end)
            busy_start = max(window_start, busy_end - timedelta(hours=duration_hours))
            if busy_end <= busy_start:
                continue
            for bucket in buckets:
                overlap_start = max(busy_start, bucket["start"])
                overlap_end = min(busy_end, bucket["end"])
                if overlap_end <= overlap_start:
                    continue
                overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600.0
                index = bucket["index"]
                area_counts[block_id][index] += count * overlap_hours / duration_hours
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
    block_bucket_indices = (1, 2, 3)
    hard_block_threshold = 0.3
    vessel_overlap_hard_block_threshold = 0.2
    for block_id in sorted(set(area_busy_hours) | set(area_vessel_overlap_blocks)):
        busy_hours = area_busy_hours[block_id]
        ratios = [min(1.0, hours / bucket_hours) for hours in busy_hours]
        peak_ratio = max(
            (
                ratios[index]
                for index in block_bucket_indices
                if index < len(ratios)
            ),
            default=0.0,
        )
        busy_ratio_blocked = peak_ratio > hard_block_threshold
        overlap_blocks = area_vessel_overlap_blocks.get(block_id, [])
        vessel_overlap_blocked = (
            bool(overlap_blocks)
            and peak_ratio > vessel_overlap_hard_block_threshold
        )
        planning_peak_ratio = peak_ratio
        if overlap_blocks and peak_ratio <= vessel_overlap_hard_block_threshold:
            planning_peak_ratio = 0.0
        capacity_factor = max(0.0, 1.0 - planning_peak_ratio)
        hard_blocked = busy_ratio_blocked or vessel_overlap_blocked
        areas[block_id] = {
            "bucketContainerCounts": [
                round(value, 1) for value in area_counts[block_id]
            ],
            "bucketBusyHours": [round(value, 3) for value in busy_hours],
            "bucketBusyRatios": [round(value, 4) for value in ratios],
            "peakBusyRatio": round(peak_ratio, 4),
            "planningPeakBusyRatio": round(planning_peak_ratio, 4),
            "capacityFactor": round(capacity_factor, 4),
            "hardBlocked": hard_blocked,
            "busyRatioBlocked": busy_ratio_blocked,
            "vesselOverlapBlocked": vessel_overlap_blocked,
            "vesselOverlapBlocks": overlap_blocks,
            "hardBlockBucketIndices": list(block_bucket_indices),
            "hardBlockThreshold": hard_block_threshold,
            "vesselOverlapHardBlockThreshold": vessel_overlap_hard_block_threshold,
        }


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
        "hardBlockBucketIndices": list(block_bucket_indices),
        "hardBlockThreshold": hard_block_threshold,
        "vesselOverlapHardBlockThreshold": vessel_overlap_hard_block_threshold,
        "coreReceivingStart": core_receiving_start.isoformat(),
        "coreReceivingEnd": core_receiving_end.isoformat(),
    }


def run_plan(
    token: Optional[str] = None,
    *,
    line_keys: Any = None,
    vessel_key: Any = None,
    type: Optional[int] = None,
    vessel_line_keys: Any = None,
    vessel_visit_keys: Any = None,
    import_info: Optional[Mapping[str, Any]] = None,
    export_info: Optional[Mapping[str, Any]] = None,
    is_auto_group: Optional[int] = None,
    is_auto_range: Optional[int] = None,
    apply_to_yard: bool = False,
    save_visualization: bool = False,
    visualization_path: Optional[str] = None,
    visualization_block_ids: Optional[Sequence[str]] = None,
    visualization_use_real_coordinates: bool = True,
    plan_start_time: Optional[datetime] = None,
    plan_end_time: Optional[datetime] = None,
    return_list: bool = False,
    print_score: bool = False,
) -> Union[PlanningResult, List[Dict[str, Any]], Dict[str, Any]]:
    """
    规划入口：传入单个航次或航线，完成进出口箱堆场分配规划。

    token: 第一个参数，TOS/平台访问令牌（str），默认 None；不传时由调用方（如 FastAPI）决定是否回退到服务注册 token。
    vessel_line_keys: 对应请求体 vesselLineKeyList，当前只支持一个元素。
    vessel_visit_keys: 对应请求体 vesselVisitKeyList，当前只支持一个元素；与 vessel_line_keys 二选一。
    import_info / export_info: 请求中的进出口数量、POD 和允许箱区。
    is_auto_group / is_auto_range: 两个字段都传 0/1，且必须恰好一个为 1；
        分别兼容旧 type=1 和 type=2。line_keys、vessel_key、type 保留给旧调用方。
    return_list: 为 True 时直接返回 API 用的分配组列表（等同 metrics["range_plan"]["data"]）；
        为 False 时返回完整 PlanningResult（本地调试、可视化用）。

    plan_start_time / plan_end_time: 手动规划范围；仅纳入与该区间有交集的航次，
    滚动时间步亦使用该区间。二者须同时指定或同时省略。
    """

    plan_type = _resolve_plan_type(
        legacy_type=type,
        is_auto_group=is_auto_group,
        is_auto_range=is_auto_range,
    )
    demand = _normalize_demand_config(import_info, export_info)

    plan_start_time, plan_end_time = _normalize_plan_window(
        plan_start_time, plan_end_time
    )

    loader = TOSLoader(token=token)
    if line_keys is not None and vessel_line_keys is not None:
        raise ValueError("Use vessel_line_keys instead of line_keys, not both")
    if vessel_key is not None and vessel_visit_keys is not None:
        raise ValueError("Use vessel_visit_keys instead of vessel_key, not both")

    raw_line_keys = vessel_line_keys if vessel_line_keys is not None else line_keys
    raw_vessel_keys = vessel_visit_keys if vessel_visit_keys is not None else vessel_key
    normalized_line_key = _normalize_single_key(loader, raw_line_keys, "lineKey")
    normalized_vessel_key = _normalize_single_key(
        loader, raw_vessel_keys, "VesselVisit dbkey"
    )

    has_line_keys = normalized_line_key is not None
    has_vessel_key = normalized_vessel_key is not None
    if has_line_keys == has_vessel_key:
        raise ValueError("Provide exactly one of line_keys or vessel_key")

    print("=" * 70)
    if has_vessel_key:
        print(f"  visitDbkey: {normalized_vessel_key}")
    print(f"  堆场规划  type={plan_type}  lineKey: {normalized_line_key}")
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
    containers, demand_metrics = _augment_containers_to_minimum_demands(
        containers=import_containers + export_containers,
        vessels=vessels,
        line_key=normalized_line_key,
        demand=demand,
        random_seed=(
            f"line={normalized_line_key}|visit={normalized_vessel_key}|"
            f"type={plan_type}|demand={demand!r}"
        ),
    )
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

    from useable_space import YardSpace

    yard = YardSpace.load(token=token)
    block_business_types, selected_block_ids, allowed_blocks_by_business = (
        _resolve_block_scope(yard, demand)
    )
    if plan_type == 2:
        range_plan = loader.build_type2_space_allocation_plan(
            containers=containers,
            yard=yard,
            allowed_block_ids_by_business=(
                allowed_blocks_by_business if demand is not None else None
            ),
        )
        result = PlanningResult(
            run_id=f"PLAN-TYPE2-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            mode=PlannerMode.FULL_PLAN,
        )
        result.metrics["range_plan"] = range_plan
        result.metrics["data"] = range_plan.get("data", [])
        result.metrics["demand"] = demand_metrics

        changed_groups = range_plan.get("groupMap", {})
        api_items = range_plan.get("data", [])
        print(f"\n  type=2 groups with appended ranges: {len(changed_groups)}")
        for group_key, group in list(changed_groups.items())[:20]:
            ranges = group.get("rangeList") or []
            print(
                f"    groupKey={group_key} groupName={group.get('groupName')} "
                f"rangeCount={len(ranges)}"
            )
        if len(changed_groups) > 20:
            print(f"    ... total {len(changed_groups)} groups")

        if return_list:
            return api_items
        return result

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
    )
    yard.print_summary()

    planner = YardPlanner()
    result = planner.plan_with_yard_space(
        yard=yard,
        containers=containers,
        block_business_types=block_business_types,
        block_ids=selected_block_ids,
        vessels=vessels,
        yard_busy_profile=yard_busy_profile,
        mode=PlannerMode.FULL_PLAN,
        apply_to_yard=apply_to_yard,
        horizon_start=horizon_start,
        horizon_end=horizon_end,
        print_score=print_score,
    )

    result.metrics["yard_cleanup"] = cleanup_metrics
    result.metrics["yard_busy_profile"] = yard_busy_profile
    result.metrics["demand"] = demand_metrics

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
