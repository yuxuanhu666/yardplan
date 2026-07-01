from __future__ import annotations

import argparse
import ast
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from yardplan_core.simultaneous import (
    simultaneous_loading_conflict_group_ids,
    simultaneous_loading_safety_gap_bays,
)


@dataclass(frozen=True)
class AllocationFact:
    root_group_id: str
    group_id: str
    yard_area_id: str
    planned_columns: float
    planned_containers: float
    business_type: Any
    group: Any
    allocation: Any


def score_yard_plan(
    result: Any,
    *,
    yard_areas: Optional[Sequence[Any]] = None,
    workload_snapshot: Optional[Any] = None,
    vessels: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Score one yard-planning result with five 10-point indicators.

    The function accepts the native PlanningResult object. It deliberately uses
    duck typing so it can also tolerate compatible test doubles.
    """

    yard_areas = list(yard_areas or [])
    groups = list(getattr(result, "allocation_groups", []) or [])
    allocations = list(getattr(result, "bay_column_allocations", []) or [])
    assignments = list(getattr(result, "area_assignments", []) or [])
    metric_vessels = None
    metrics = getattr(result, "metrics", {}) or {}
    if isinstance(metrics, dict):
        metric_vessels = metrics.get("vessels")
    vessel_map = _normalize_vessels(vessels or metric_vessels)
    area_by_id = {str(getattr(area, "area_id", "")): area for area in yard_areas}

    group_by_id = {getattr(group, "group_id", ""): group for group in groups}
    facts = _build_allocation_facts(allocations, assignments, group_by_id)
    root_demands = _root_demands(groups, facts)

    items = {
        "transport_distance": _score_transport_distance(
            root_demands,
            facts,
            area_by_id=area_by_id,
            vessel_map=vessel_map,
        ),
        "demand_satisfaction": _score_demand_satisfaction(root_demands, facts),
        "business_dispersion": _score_business_dispersion(
            root_demands,
            facts,
            groups=groups,
            vessel_map=vessel_map,
        ),
        "area_peak_staggering": _score_area_peak_staggering(
            facts,
            yard_areas=yard_areas,
            workload_snapshot=workload_snapshot,
        ),
        "bay_quality": _score_bay_quality(
            allocations,
            group_by_id=group_by_id,
            yard_areas=yard_areas,
        ),
    }
    total = round(sum(item["score"] for item in items.values()), 2)
    return {
        "totalScore": total,
        "maxScore": 50.0,
        "items": items,
        "summary": {
            "groupCount": len(root_demands),
            "allocationCount": len(allocations),
            "areaAssignmentCount": len(assignments),
            "unassignedGroupCount": len(getattr(result, "unassigned_groups", []) or []),
            "vesselCount": len(vessel_map),
        },
    }


def format_score_report(score: Dict[str, Any], *, indent: str = "") -> str:
    labels = {
        "transport_distance": "Transport distance",
        "demand_satisfaction": "Demand satisfaction",
        "business_dispersion": "Business dispersion / concentration",
        "area_peak_staggering": "Area peak staggering",
        "bay_quality": "Bay quality / purity / rehandle risk",
    }
    lines = [
        f"{indent}Total score: {_fmt(score.get('totalScore'))}/{_fmt(score.get('maxScore', 50.0))}"
    ]
    for key, label in labels.items():
        item = (score.get("items") or {}).get(key)
        if not item:
            continue
        lines.append(
            f"{indent}- {label}: {_fmt(item.get('score'))}/{_fmt(item.get('maxScore', 10.0))} "
            f"({item.get('formula', '')})"
        )
    return "\n".join(lines)


def _score_area_peak_staggering(
    facts: Sequence[AllocationFact],
    *,
    yard_areas: Sequence[Any],
    workload_snapshot: Optional[Any],
) -> Dict[str, Any]:
    series = _build_voyage_workload_series(facts, workload_snapshot)
    if not series:
        return _item(10.0, "No cross-vessel workload to compare")

    area_voyages: DefaultDict[str, set[str]] = defaultdict(set)
    for area_id, voyage_id in series:
        area_voyages[area_id].add(voyage_id)

    weighted_score = 0.0
    total_possible = 0.0
    area_details: Dict[str, Any] = {}

    for area_id, voyages in sorted(area_voyages.items()):
        ordered_voyages = sorted(voyages)
        area_overlap = 0.0
        area_possible = 0.0
        pair_details: List[Dict[str, Any]] = []

        for left_index, left_voyage in enumerate(ordered_voyages):
            for right_voyage in ordered_voyages[left_index + 1 :]:
                left_series = series[(area_id, left_voyage)]
                right_series = series[(area_id, right_voyage)]
                pair_overlap, pair_possible, pair_steps = _opposite_direction_overlap(
                    left_series,
                    right_series,
                )
                if pair_possible <= 1e-9:
                    continue

                pair_ratio = _safe_div(pair_overlap, pair_possible)
                pair_score = 10.0 * (1.0 - _clamp(pair_ratio, 0.0, 1.0))
                area_overlap += pair_overlap
                area_possible += pair_possible
                pair_details.append(
                    {
                        "voyages": [left_voyage, right_voyage],
                        "oppositeOverlapMoves": round(pair_overlap, 3),
                        "oppositePossibleMoves": round(pair_possible, 3),
                        "conflictRatio": round(pair_ratio, 4),
                        "score": round(pair_score, 2),
                        "steps": pair_steps,
                    }
                )

        if area_possible <= 1e-9:
            area_score = 10.0
            conflict_ratio = 0.0
        else:
            conflict_ratio = _safe_div(area_overlap, area_possible)
            area_score = 10.0 * (1.0 - _clamp(conflict_ratio, 0.0, 1.0))
            weighted_score += area_possible * area_score
            total_possible += area_possible

        area_details[area_id] = {
            "voyageCount": len(ordered_voyages),
            "oppositeOverlapMoves": round(area_overlap, 3),
            "oppositePossibleMoves": round(area_possible, 3),
            "conflictRatio": round(conflict_ratio, 4),
            "score": round(area_score, 2),
            "pairs": pair_details,
        }

    if total_possible <= 1e-9:
        return _item(
            10.0,
            "No different-voyage opposite-direction overlap",
            areas=area_details,
        )
    return _item(
        weighted_score / total_possible,
        "1 - different-voyage opposite-direction overlap ratio",
        areas=area_details,
    )


def _score_bay_quality(
    allocations: Sequence[Any],
    *,
    group_by_id: Dict[str, Any],
    yard_areas: Sequence[Any],
    full_threshold: float = 0.85,
) -> Dict[str, Any]:
    bay_capacity = _bay_capacity_by_area(yard_areas)
    bay_group_loads: DefaultDict[Tuple[str, int], DefaultDict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    bay_group_sizes: DefaultDict[Tuple[str, int], Dict[str, Any]] = defaultdict(dict)
    group_total_by_bay_capacity: DefaultDict[Tuple[str, float], float] = defaultdict(float)
    column_weight_counts: DefaultDict[Tuple[str, Any, int], DefaultDict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for allocation in allocations:
        group_id = str(getattr(allocation, "group_id", ""))
        area_id = str(getattr(allocation, "yard_area_id", ""))
        if not group_id or not area_id:
            continue
        group = group_by_id.get(group_id)
        size = _enum_value(getattr(group, "size", getattr(allocation, "size", None)))
        weight_name = _weight_class_name(getattr(group, "weight_class", None))
        for bay_number, columns in _allocation_columns_by_bay(allocation).items():
            load = max(0.0, columns)
            if load <= 0.0:
                continue
            bay_key = (area_id, bay_number)
            capacity = bay_capacity.get(bay_key, load)
            group_key = (group_id, capacity)
            bay_group_loads[bay_key][group_id] += load
            bay_group_sizes[bay_key][group_id] = size
            group_total_by_bay_capacity[group_key] += load
            if bay_key not in bay_capacity:
                bay_capacity[bay_key] = max(1.0, load)
        for bay_spec, stack_index, tiers in _allocation_exact_columns(allocation):
            if not tiers:
                continue
            column_key = (area_id, _normalise_bay_spec(bay_spec), int(stack_index))
            column_weight_counts[column_key][weight_name] += len(tiers)

    total_planned = sum(
        sum(group_loads.values()) for group_loads in bay_group_loads.values()
    )
    if total_planned <= 0.0:
        return _item(0.0, "No placed bay columns")

    activated_capacity = 0.0
    planned_in_full_bays = 0.0
    purity_weighted = 0.0
    bay_details: Dict[str, Any] = {}
    for bay_key, group_loads in sorted(bay_group_loads.items()):
        planned = sum(group_loads.values())
        capacity = max(1.0, bay_capacity.get(bay_key, planned))
        utilization = _safe_div(planned, capacity)
        activated_capacity += capacity
        if utilization >= full_threshold:
            planned_in_full_bays += planned

        purity = _bay_purity_score(
            group_loads=group_loads,
            group_sizes=bay_group_sizes.get(bay_key, {}),
            group_total_by_bay_capacity=group_total_by_bay_capacity,
            bay_capacity=capacity,
        )
        purity_weighted += planned * purity
        bay_details[f"{bay_key[0]}:{bay_key[1]}"] = {
            "plannedColumns": round(planned, 3),
            "capacity": round(capacity, 3),
            "utilization": round(utilization, 4),
            "groupCount": len(group_loads),
            "purity": round(purity, 3),
        }

    full_ratio = _safe_div(planned_in_full_bays, total_planned)
    activated_util = _safe_div(total_planned, activated_capacity)
    utilization_score = 0.6 * full_ratio + 0.4 * min(
        _safe_div(activated_util, full_threshold),
        1.0,
    )
    purity_score = _safe_div(purity_weighted, total_planned)
    rehandle_score, column_details = _rehandle_risk_score(column_weight_counts)
    score = 10.0 * (
        0.45 * utilization_score
        + 0.25 * purity_score
        + 0.30 * rehandle_score
    )

    return _item(
        score,
        "0.45*bay_utilization_score + 0.25*bay_purity_score + 0.30*rehandle_risk_score",
        fullThreshold=full_threshold,
        fullRatio=round(full_ratio, 4),
        activatedUtilization=round(activated_util, 4),
        utilizationScore=round(utilization_score, 4),
        purityScore=round(purity_score, 4),
        rehandleRiskScore=round(rehandle_score, 4),
        rehandleRiskRaw=round(10.0 * rehandle_score, 2),
        plannedColumns=round(total_planned, 3),
        bays=bay_details,
        columns=column_details,
    )


def _bay_purity_score(
    *,
    group_loads: Dict[str, float],
    group_sizes: Dict[str, Any],
    group_total_by_bay_capacity: Dict[Tuple[str, float], float],
    bay_capacity: float,
) -> float:
    if len(group_loads) <= 1:
        return 1.0

    sizes = {str(size) for size in group_sizes.values() if size is not None}
    if len(sizes) != 1:
        return 0.0

    for group_id, load in group_loads.items():
        total_for_capacity = group_total_by_bay_capacity.get((group_id, bay_capacity), 0.0)
        tail = total_for_capacity % bay_capacity
        if tail <= 1e-6:
            return 0.0
        if load > tail + 1e-6:
            return 0.0
    return 0.8


def _bay_capacity_by_area(yard_areas: Sequence[Any]) -> Dict[Tuple[str, int], float]:
    capacities: Dict[Tuple[str, int], float] = {}
    for area in yard_areas:
        area_id = str(getattr(area, "area_id", ""))
        for bay in getattr(area, "bays", []) or []:
            bay_number = int(getattr(bay, "bay_number", 0))
            capacities[(area_id, bay_number)] = max(
                1.0,
                _as_float(getattr(bay, "total_columns", 0.0)),
            )
    return capacities


def _score_transport_distance(
    root_demands: Dict[str, float],
    facts: Sequence[AllocationFact],
    *,
    area_by_id: Dict[str, Any],
    vessel_map: Dict[str, Any],
) -> Dict[str, Any]:
    if not root_demands:
        return _item(0.0, "No demand groups")

    root_groups: Dict[str, Any] = {}
    by_root_area: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for fact in facts:
        by_root_area[fact.root_group_id][fact.yard_area_id] += max(0.0, fact.planned_containers)
        if fact.root_group_id not in root_groups and fact.group is not None:
            root_groups[fact.root_group_id] = fact.group

    total_demand = sum(root_demands.values())
    weighted = 0.0
    details: Dict[str, Any] = {}
    for root_id, demand in root_demands.items():
        group = root_groups.get(root_id)
        voyage_id = str(getattr(group, "voyage_id", "") or "")
        area_loads = by_root_area.get(root_id, {})
        assigned = sum(area_loads.values())
        if assigned <= 0.0:
            details[root_id] = {"score": 0.0, "reason": "unassigned", "voyageId": voyage_id}
            continue

        distance_rows: List[Tuple[str, float, float]] = []
        for area_id, load in area_loads.items():
            distance = _area_distance_to_berth(
                area_by_id=area_by_id,
                vessel_map=vessel_map,
                area_id=area_id,
                voyage_id=voyage_id,
            )
            if distance is None:
                continue
            distance_rows.append((area_id, load, distance))

        if not distance_rows:
            details[root_id] = {
                "score": 0.0,
                "reason": "missing distance data",
                "voyageId": voyage_id,
            }
            continue

        weighted_distance = sum(load * dist for _area, load, dist in distance_rows) / max(
            1e-9, sum(load for _area, load, _dist in distance_rows)
        )
        ordered = sorted(distance_rows, key=lambda item: (item[2], item[0]))
        anchor_count = _distance_anchor_count(len(ordered))
        best = sum(dist for _area, _load, dist in ordered[:anchor_count]) / anchor_count
        worst = sum(dist for _area, _load, dist in ordered[-anchor_count:]) / anchor_count
        if worst <= best + 1e-9:
            group_score = 10.0
        else:
            group_score = 10.0 * _clamp(
                (worst - weighted_distance) / (worst - best),
                0.0,
                1.0,
            )

        weighted += (demand / total_demand) * group_score
        details[root_id] = {
            "score": round(group_score, 2),
            "voyageId": voyage_id,
            "assignedAreas": len(distance_rows),
            "anchorCount": anchor_count,
            "bestReference": round(best, 3),
            "worstReference": round(worst, 3),
            "weightedDistance": round(weighted_distance, 3),
        }

    return _item(
        10.0 * weighted,
        "relative weighted distance between nearest and farthest assigned areas",
        groups=details,
    )


def _score_business_dispersion(
    root_demands: Dict[str, float],
    facts: Sequence[AllocationFact],
    *,
    groups: Sequence[Any],
    vessel_map: Dict[str, Any],
) -> Dict[str, Any]:
    if not root_demands:
        return _item(0.0, "No demand groups")

    root_groups: Dict[str, Any] = {}
    for group in groups:
        group_id = str(getattr(group, "group_id", "") or "")
        if not group_id:
            continue
        root_id = _root_group_id(group, group_id)
        if getattr(group, "parent_group_id", None):
            root_groups.setdefault(root_id, group)
        else:
            root_groups[root_id] = group

    voyage_area_loads: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    root_area_bays: DefaultDict[str, DefaultDict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    voyage_demands: DefaultDict[str, float] = defaultdict(float)
    voyage_groups: Dict[str, Any] = {}
    for fact in facts:
        group = fact.group
        voyage_id = str(getattr(group, "voyage_id", "") or "")
        if not voyage_id:
            continue
        voyage_area_loads[voyage_id][fact.yard_area_id] += max(0.0, fact.planned_containers)
        root_area_bays[fact.root_group_id][fact.yard_area_id].update(
            _allocation_bays(fact.allocation)
        )
        if voyage_id not in voyage_groups and group is not None:
            voyage_groups[voyage_id] = group
        if fact.root_group_id not in root_groups and group is not None:
            root_groups[fact.root_group_id] = group

    voyage_root_ids: DefaultDict[str, set[str]] = defaultdict(set)
    for root_id, demand in root_demands.items():
        group = root_groups.get(root_id)
        voyage_id = str(getattr(group, "voyage_id", "") or "")
        if not voyage_id:
            continue
        voyage_demands[voyage_id] += max(0.0, demand)
        voyage_groups.setdefault(voyage_id, group)
        voyage_root_ids[voyage_id].add(root_id)

    total_demand = sum(voyage_demands.values())
    if total_demand <= 0.0:
        return _item(0.0, "No voyage demand groups")

    weighted = 0.0
    details: Dict[str, Any] = {}
    for voyage_id, demand in sorted(voyage_demands.items()):
        group = voyage_groups.get(voyage_id)
        vessel = vessel_map.get(voyage_id)
        area_loads = voyage_area_loads.get(voyage_id, {})
        assigned = sum(area_loads.values())
        if assigned <= 0.0:
            details[voyage_id] = {
                "score": 0.0,
                "reason": "unassigned",
                "voyageId": voyage_id,
            }
            continue

        assigned_areas = len([v for v in area_loads.values() if v > 0.0])
        eqp_num = _vessel_eqp_num(vessel)
        if eqp_num <= 0:
            eqp_num = _group_eqp_num(group)

        area_match_raw = 0.0
        if eqp_num <= 0:
            status = "missing crane count"
            target_areas = None
        else:
            target_areas = eqp_num
            area_match_raw = _voyage_area_match_score(
                assigned_areas=assigned_areas,
                target_areas=target_areas,
            )
            status = "target=crane_count, accept [-1,+2]"

        sim_safety_raw, pair_details = _voyage_simultaneous_loading_safety_score(
            voyage_id=voyage_id,
            root_ids=voyage_root_ids.get(voyage_id, set()),
            root_groups=root_groups,
            root_area_bays=root_area_bays,
        )
        vessel_score = 0.6 * area_match_raw + 0.4 * sim_safety_raw
        violating_pair_count = sum(
            1 for pair in pair_details if pair.get("score", 10.0) < 10.0
        )
        gap_values = [
            int(pair["minGap"])
            for pair in pair_details
            if pair.get("minGap") is not None
        ]

        weighted += (demand / total_demand) * vessel_score
        details[voyage_id] = {
            "score": round(vessel_score, 2),
            "voyageId": voyage_id,
            "cranes": eqp_num,
            "targetAreas": target_areas,
            "assignedAreas": assigned_areas,
            "groupDemand": round(demand, 3),
            "areaMatchRaw": round(area_match_raw, 2),
            "simSafetyRaw": round(sim_safety_raw, 2),
            "violatingPairCount": violating_pair_count,
            "worstGap": min(gap_values) if gap_values else None,
            "pairs": pair_details,
            "status": status,
        }

    return _item(
        10.0 * weighted,
        "0.6*area_match_raw + 0.4*sim_safety_raw",
        voyages=details,
    )


def _normalize_vessels(vessels: Optional[Any]) -> Dict[str, Any]:
    if vessels is None:
        return {}
    if isinstance(vessels, dict):
        return {str(key): value for key, value in vessels.items()}

    normalized: Dict[str, Any] = {}
    for vessel in vessels:
        voyage_id = str(getattr(vessel, "voyage_id", "") or "")
        vessel_id = str(getattr(vessel, "vessel_id", "") or "")
        key = voyage_id or vessel_id
        if key:
            normalized[key] = vessel
    return normalized


def _group_eqp_num(group: Any) -> int:
    if group is None:
        return 0

    attrs = getattr(group, "group_attributes", {}) or {}
    for key in ("eqp_num", "EQP_NUM", "crane_num", "craneCount"):
        value = attrs.get(key)
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count

    value = getattr(group, "eqp_num", None)
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, count)


def _vessel_eqp_num(vessel: Any) -> int:
    if vessel is None:
        return 0
    try:
        count = int(getattr(vessel, "eqp_num", 0) or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, count)


def _voyage_area_match_score(*, assigned_areas: int, target_areas: int) -> float:
    if target_areas <= 0:
        return 0.0
    if assigned_areas < target_areas - 1 or assigned_areas > target_areas + 2:
        return 0.0
    return max(0.0, 10.0 - 2.0 * abs(assigned_areas - target_areas))


def _voyage_simultaneous_loading_safety_score(
    *,
    voyage_id: str,
    root_ids: set[str],
    root_groups: Dict[str, Any],
    root_area_bays: DefaultDict[str, DefaultDict[str, set[int]]],
) -> Tuple[float, List[Dict[str, Any]]]:
    del voyage_id
    pair_keys: set[Tuple[str, str]] = set()
    pair_details: List[Dict[str, Any]] = []

    for root_id in sorted(root_ids):
        group = root_groups.get(root_id)
        if group is None:
            continue
        for other_root_id in simultaneous_loading_conflict_group_ids(group):
            if other_root_id not in root_ids or other_root_id == root_id:
                continue
            pair_key = tuple(sorted((root_id, other_root_id)))
            if pair_key in pair_keys:
                continue
            pair_keys.add(pair_key)

            left_group = root_groups.get(pair_key[0])
            right_group = root_groups.get(pair_key[1])
            default_gap = 4
            required_gap = max(
                int(simultaneous_loading_safety_gap_bays(left_group, default_gap) or 0),
                int(simultaneous_loading_safety_gap_bays(right_group, default_gap) or 0),
                default_gap,
            )
            shared_areas = sorted(
                set(root_area_bays.get(pair_key[0], {}))
                & set(root_area_bays.get(pair_key[1], {}))
            )
            min_gap: Optional[int] = None
            if shared_areas:
                for area_id in shared_areas:
                    left_bays = root_area_bays.get(pair_key[0], {}).get(area_id, set())
                    right_bays = root_area_bays.get(pair_key[1], {}).get(area_id, set())
                    area_gap = _min_bay_gap(left_bays, right_bays)
                    if min_gap is None or area_gap < min_gap:
                        min_gap = area_gap

            pair_score = _simultaneous_gap_score(min_gap, required_gap)
            pair_details.append(
                {
                    "pair": [pair_key[0], pair_key[1]],
                    "requiredGap": required_gap,
                    "sharedAreas": shared_areas,
                    "minGap": min_gap,
                    "score": round(pair_score, 2),
                }
            )

    if not pair_details:
        return 10.0, []

    average_score = sum(float(pair["score"]) for pair in pair_details) / len(pair_details)
    return average_score, pair_details


def _area_distance_to_berth(
    *,
    area_by_id: Dict[str, Any],
    vessel_map: Dict[str, Any],
    area_id: str,
    voyage_id: str,
) -> Optional[float]:
    area = area_by_id.get(area_id)
    if area is None:
        return None

    distance_map = getattr(area, "distance_to_berth", {}) or {}
    if voyage_id and voyage_id in distance_map:
        return _as_float(distance_map.get(voyage_id))

    vessel = vessel_map.get(voyage_id) if voyage_id else None
    if vessel is None:
        return None

    area_coord = getattr(area, "center_coordinate", None)
    berth_coord = getattr(vessel, "berth_coordinate", None)
    if area_coord is None or berth_coord is None:
        return None
    return math.hypot(
        float(area_coord[0]) - float(berth_coord[0]),
        float(area_coord[1]) - float(berth_coord[1]),
    )


def _distance_anchor_count(area_count: int) -> int:
    if area_count <= 1:
        return area_count
    if area_count <= 3:
        return 1
    return min(3, max(1, area_count // 2))


def _build_allocation_facts(
    allocations: Sequence[Any],
    assignments: Sequence[Any],
    group_by_id: Dict[str, Any],
) -> List[AllocationFact]:
    facts: List[AllocationFact] = []
    assignment_demand_by_group_area: Dict[Tuple[str, str], float] = defaultdict(float)
    for assignment in assignments:
        group_id = str(getattr(assignment, "group_id", ""))
        area_id = str(getattr(assignment, "yard_area_id", ""))
        assignment_demand_by_group_area[(group_id, area_id)] += max(
            0.0,
            _as_float(getattr(assignment, "column_demand", 0.0)),
        )

    for allocation in allocations:
        group_id = str(getattr(allocation, "group_id", ""))
        area_id = str(getattr(allocation, "yard_area_id", ""))
        if not group_id or not area_id:
            continue

        group = group_by_id.get(group_id)
        root_id = _root_group_id(group, group_id)
        planned_columns = _allocation_columns(allocation)
        if planned_columns <= 0.0:
            planned_columns = assignment_demand_by_group_area.get((group_id, area_id), 0.0)
        planned_containers = _estimate_container_count(group, planned_columns)
        facts.append(
            AllocationFact(
                root_group_id=root_id,
                group_id=group_id,
                yard_area_id=area_id,
                planned_columns=planned_columns,
                planned_containers=planned_containers,
                business_type=getattr(group, "business_type", getattr(allocation, "business_type", None)),
                group=group,
                allocation=allocation,
            )
        )
    return facts


def _root_demands(groups: Sequence[Any], facts: Sequence[AllocationFact]) -> Dict[str, float]:
    demands: Dict[str, float] = {}
    for group in groups:
        group_id = str(getattr(group, "group_id", ""))
        if not group_id or getattr(group, "parent_group_id", None):
            continue
        demand = _group_container_demand(group)
        if demand > 0.0:
            demands[group_id] = demand

    for fact in facts:
        if fact.root_group_id not in demands:
            demands[fact.root_group_id] = max(0.0, fact.planned_containers)
    return demands


def _score_demand_satisfaction(
    root_demands: Dict[str, float],
    facts: Sequence[AllocationFact],
) -> Dict[str, Any]:
    if not root_demands:
        return _item(0.0, "No demand groups")

    assigned_by_root: DefaultDict[str, float] = defaultdict(float)
    for fact in facts:
        assigned_by_root[fact.root_group_id] += fact.planned_containers

    total_demand = sum(root_demands.values())
    weighted = 0.0
    complete_count = 0
    for root_id, demand in root_demands.items():
        assigned = min(demand, assigned_by_root.get(root_id, 0.0))
        ratio = _safe_div(assigned, demand)
        complete = 1.0 if assigned + 1e-6 >= demand else 0.0
        complete_count += int(complete)
        weighted += (demand / total_demand) * (0.7 * ratio + 0.3 * complete)

    return _item(
        10.0 * weighted,
        "0.7*assigned_ratio + 0.3*complete_group_ratio",
        demand=sum(root_demands.values()),
        assigned=round(sum(min(root_demands[k], assigned_by_root.get(k, 0.0)) for k in root_demands), 3),
        completeGroups=complete_count,
        totalGroups=len(root_demands),
    )


def _build_workload_series(
    facts: Sequence[AllocationFact],
    yard_areas: Sequence[Any],
    workload_snapshot: Optional[Any],
) -> Dict[Tuple[str, int], Dict[str, float]]:
    area_ids = {str(getattr(area, "area_id", "")) for area in yard_areas if getattr(area, "area_id", "")}
    area_ids.update(fact.yard_area_id for fact in facts if fact.yard_area_id)

    steps = list(getattr(workload_snapshot, "steps", []) or [])
    step_ids = [int(getattr(step, "step_id", idx)) for idx, step in enumerate(steps)]
    if not step_ids and facts:
        step_ids = [0]

    fallback_capacity = _fallback_max_moves(workload_snapshot)
    series: Dict[Tuple[str, int], Dict[str, float]] = {}
    for area_id in sorted(area_ids):
        for step_id in step_ids:
            base = None
            if workload_snapshot is not None:
                base = getattr(workload_snapshot, "by_area_step", {}).get((area_id, step_id))
            inbound = _as_float(getattr(base, "inbound_moves", 0.0)) if base is not None else 0.0
            outbound = _as_float(getattr(base, "outbound_moves", 0.0)) if base is not None else 0.0
            max_moves = _as_float(getattr(base, "max_moves", fallback_capacity)) if base is not None else fallback_capacity
            series[(area_id, step_id)] = {
                "inbound": inbound,
                "outbound": outbound,
                "total": inbound + outbound,
                "max_moves": max(1.0, max_moves),
            }

    for fact in facts:
        fact_steps = _step_ids_for_group(fact.group, workload_snapshot)
        if not fact_steps:
            fact_steps = step_ids or [0]
        per_step = fact.planned_containers / max(1, len(fact_steps))
        inbound_delta, outbound_delta = _move_delta(fact.business_type, per_step)
        for step_id in fact_steps:
            key = (fact.yard_area_id, int(step_id))
            if key not in series:
                series[key] = {"inbound": 0.0, "outbound": 0.0, "total": 0.0, "max_moves": fallback_capacity}
            series[key]["inbound"] += inbound_delta
            series[key]["outbound"] += outbound_delta
            series[key]["total"] = series[key]["inbound"] + series[key]["outbound"]
    return series


def _build_voyage_workload_series(
    facts: Sequence[AllocationFact],
    workload_snapshot: Optional[Any],
) -> Dict[Tuple[str, str], Dict[int, Dict[str, float]]]:
    series: DefaultDict[Tuple[str, str], DefaultDict[int, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"inbound": 0.0, "outbound": 0.0})
    )
    for fact in facts:
        group = fact.group
        voyage_id = str(getattr(group, "voyage_id", "") or "")
        if not voyage_id:
            continue

        fact_steps = _step_ids_for_group(group, workload_snapshot)
        if not fact_steps and workload_snapshot is not None:
            step_id = getattr(workload_snapshot, "step_for_voyage", lambda _voyage: None)(
                voyage_id
            )
            if step_id is not None:
                fact_steps = [int(step_id)]
        if not fact_steps:
            fact_steps = [0]

        per_step = fact.planned_containers / max(1, len(fact_steps))
        inbound_delta, outbound_delta = _move_delta(fact.business_type, per_step)
        for step_id in fact_steps:
            row = series[(fact.yard_area_id, voyage_id)][int(step_id)]
            row["inbound"] += inbound_delta
            row["outbound"] += outbound_delta

    return {
        key: {step_id: dict(values) for step_id, values in rows.items()}
        for key, rows in series.items()
    }


def _opposite_direction_overlap(
    left_series: Dict[int, Dict[str, float]],
    right_series: Dict[int, Dict[str, float]],
) -> Tuple[float, float, List[Dict[str, Any]]]:
    overlap = 0.0
    possible = 0.0
    step_details: List[Dict[str, Any]] = []
    for step_id in sorted(set(left_series).intersection(right_series)):
        left = left_series.get(step_id, {})
        right = right_series.get(step_id, {})
        left_inbound = max(0.0, _as_float(left.get("inbound")))
        left_outbound = max(0.0, _as_float(left.get("outbound")))
        right_inbound = max(0.0, _as_float(right.get("inbound")))
        right_outbound = max(0.0, _as_float(right.get("outbound")))

        step_overlap = min(left_inbound, right_outbound) + min(left_outbound, right_inbound)
        step_possible = min(
            left_inbound + left_outbound,
            right_inbound + right_outbound,
        )
        if step_possible <= 1e-9:
            continue

        overlap += step_overlap
        possible += step_possible
        step_details.append(
            {
                "stepId": int(step_id),
                "leftInbound": round(left_inbound, 3),
                "leftOutbound": round(left_outbound, 3),
                "rightInbound": round(right_inbound, 3),
                "rightOutbound": round(right_outbound, 3),
                "oppositeOverlapMoves": round(step_overlap, 3),
                "oppositePossibleMoves": round(step_possible, 3),
            }
        )

    return overlap, possible, step_details


def _step_ids_for_group(group: Any, workload_snapshot: Optional[Any]) -> List[int]:
    if group is None or workload_snapshot is None:
        return []
    steps = list(getattr(workload_snapshot, "steps", []) or [])
    if not steps:
        return []

    eta = getattr(group, "earliest_arrival", None)
    etd = getattr(group, "latest_departure", None)
    if eta is None:
        containers = list(getattr(group, "containers", []) or [])
        etas = [getattr(container, "eta", None) for container in containers if getattr(container, "eta", None)]
        eta = min(etas) if etas else None
    if etd is None:
        containers = list(getattr(group, "containers", []) or [])
        etds = [getattr(container, "etd", None) for container in containers if getattr(container, "etd", None)]
        etd = max(etds) if etds else None

    if eta is not None and etd is not None and etd > eta:
        span = [
            int(getattr(step, "step_id", index))
            for index, step in enumerate(steps)
            if getattr(step, "start_time", None) < etd and getattr(step, "end_time", None) > eta
        ]
        if span:
            return span

    voyage_id = getattr(group, "voyage_id", None)
    if voyage_id is not None and hasattr(workload_snapshot, "step_for_voyage"):
        step_id = workload_snapshot.step_for_voyage(voyage_id)
        if step_id is not None:
            return [int(step_id)]
    return []


def _allocation_columns(allocation: Any) -> float:
    total = 0.0
    for _bay_spec, columns in getattr(allocation, "bay_column_details", []) or []:
        total += max(0.0, _as_float(columns))
    for _bay_spec, start_stack, end_stack in getattr(allocation, "bay_stack_details", []) or []:
        total += max(0.0, _as_float(end_stack) - _as_float(start_stack) + 1.0)
    return total


def _allocation_columns_by_bay(allocation: Any) -> Dict[int, float]:
    by_bay: DefaultDict[int, float] = defaultdict(float)
    for bay_spec, columns in getattr(allocation, "bay_column_details", []) or []:
        for bay_number in _bay_numbers(bay_spec):
            by_bay[bay_number] += max(0.0, _as_float(columns))
    for bay_spec, start_stack, end_stack in getattr(allocation, "bay_stack_details", []) or []:
        columns = max(0.0, _as_float(end_stack) - _as_float(start_stack) + 1.0)
        for bay_number in _bay_numbers(bay_spec):
            by_bay[bay_number] += columns
    return dict(by_bay)


def _allocation_bays(allocation: Any) -> List[int]:
    bays: set[int] = set()
    for bay_spec, _columns in getattr(allocation, "bay_column_details", []) or []:
        bays.update(_bay_numbers(bay_spec))
    for bay_spec, _start_stack, _end_stack in getattr(allocation, "bay_stack_details", []) or []:
        bays.update(_bay_numbers(bay_spec))
    return sorted(bays)


def _bay_numbers(bay_spec: Any) -> List[int]:
    if isinstance(bay_spec, (list, tuple)):
        return [int(value) for value in bay_spec]
    return [int(bay_spec)]


def _min_bay_gap(left_bays: Iterable[int], right_bays: Iterable[int]) -> int:
    left_values = sorted({int(value) for value in left_bays})
    right_values = sorted({int(value) for value in right_bays})
    if not left_values or not right_values:
        return 0

    min_gap: Optional[int] = None
    for left in left_values:
        for right in right_values:
            gap = abs(int(right) - int(left))
            if min_gap is None or gap < min_gap:
                min_gap = gap
    return int(min_gap or 0)


def _simultaneous_gap_score(min_gap: Optional[int], required_gap: int) -> float:
    if min_gap is None or min_gap >= required_gap:
        return 10.0
    if min_gap == 3:
        return 7.0
    if min_gap == 2:
        return 4.0
    if min_gap == 1:
        return 1.0
    return 0.0


def _allocation_exact_columns(
    allocation: Any,
) -> List[Tuple[Any, int, List[int]]]:
    exact = _extract_exact_note(getattr(allocation, "notes", "") or "")
    if not exact:
        return []

    entries: List[Tuple[Any, int, List[int]]] = []
    for entry in exact.split("; "):
        parsed = _parse_exact_entry(entry)
        if parsed is None:
            continue
        bay_spec, stack_index, tier_ranges = parsed
        tiers: List[int] = []
        for start, end in tier_ranges:
            tiers.extend(range(int(start), int(end) + 1))
        if tiers:
            entries.append((bay_spec, int(stack_index), sorted(set(tiers))))
    return entries


def _extract_exact_note(notes: str) -> str:
    marker = "exact "
    if not notes or marker not in notes:
        return ""
    return notes.split(marker, 1)[1].strip()


def _parse_exact_entry(
    entry: str,
) -> Optional[Tuple[Any, int, List[Tuple[int, int]]]]:
    prefix, sep, tiers_text = entry.partition(" tiers ")
    if not sep:
        return None
    bay_text, sep, stack_text = prefix.partition(": stack ")
    if not sep:
        return None
    try:
        bay_spec = ast.literal_eval(bay_text.strip())
    except (SyntaxError, ValueError):
        try:
            bay_spec = int(bay_text.strip())
        except ValueError:
            return None
    try:
        stack_index = int(stack_text.strip())
    except ValueError:
        return None
    tier_ranges = _parse_tier_ranges(tiers_text.strip())
    if not tier_ranges:
        return None
    return bay_spec, stack_index, tier_ranges


def _parse_tier_ranges(text: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if "-" in piece:
            left, right = piece.split("-", 1)
            try:
                start, end = int(left), int(right)
            except ValueError:
                continue
        else:
            try:
                start = end = int(piece)
            except ValueError:
                continue
        ranges.append((min(start, end), max(start, end)))
    return ranges


def _normalise_bay_spec(bay_spec: Any) -> Any:
    if isinstance(bay_spec, (list, tuple)):
        return tuple(int(value) for value in bay_spec)
    return int(bay_spec)


def _weight_class_name(weight_class: Any) -> str:
    raw = getattr(weight_class, "value", weight_class)
    return str(raw or "light").lower()


def _rehandle_risk_score(
    column_weight_counts: DefaultDict[Tuple[str, Any, int], DefaultDict[str, int]],
) -> Tuple[float, Dict[str, Any]]:
    total_weight = 0.0
    weighted_score = 0.0
    column_details: Dict[str, Any] = {}

    for column_key, counts in sorted(
        column_weight_counts.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1]), int(item[0][2])),
    ):
        empty_count = int(counts.get("empty", 0))
        light_count = int(counts.get("light", 0))
        heavy_count = int(counts.get("heavy", 0))
        total = empty_count + light_count + heavy_count
        if total <= 0:
            continue

        numerator = (
            math.factorial(empty_count)
            * math.factorial(light_count)
            * math.factorial(heavy_count)
        )
        denominator = math.factorial(total)
        no_rehandle_probability = _safe_div(float(numerator), float(denominator))
        column_score = 10.0 * no_rehandle_probability

        total_weight += total
        weighted_score += total * no_rehandle_probability
        area_id, bay_spec, stack_index = column_key
        column_details[f"{area_id}:{bay_spec}:{stack_index}"] = {
            "emptyCount": empty_count,
            "lightCount": light_count,
            "heavyCount": heavy_count,
            "containerCount": total,
            "noRehandleProbability": round(no_rehandle_probability, 6),
            "score": round(column_score, 2),
        }

    if total_weight <= 0.0:
        return 1.0, {}
    return weighted_score / total_weight, column_details


def _group_container_demand(group: Any) -> float:
    containers = list(getattr(group, "containers", []) or [])
    if containers:
        return float(len(containers))
    count = _as_float(getattr(group, "container_count", 0.0))
    if count > 0.0:
        return count
    columns = _as_float(getattr(group, "column_demand", 0.0))
    size = getattr(group, "size", None)
    teu_factor = 2.0 if str(getattr(size, "value", size)) in {"40", "45"} else 1.0
    return max(0.0, columns / teu_factor)


def _estimate_container_count(group: Any, planned_columns: float) -> float:
    if group is None:
        return max(0.0, planned_columns)
    demand_columns = _as_float(getattr(group, "column_demand", 0.0))
    demand_containers = _group_container_demand(group)
    if demand_columns > 0.0 and demand_containers > 0.0:
        return max(0.0, demand_containers * planned_columns / demand_columns)
    return max(0.0, planned_columns)


def _root_group_id(group: Any, fallback: str) -> str:
    if group is None:
        return fallback
    return str(getattr(group, "parent_group_id", None) or getattr(group, "group_id", fallback))


def _target_area_count(demand: float) -> int:
    if demand <= 20:
        return 1
    if demand <= 80:
        return 2
    return min(4, int(math.ceil(demand / 80.0)))


def _move_delta(business_type: Any, n_containers: float) -> Tuple[float, float]:
    move_delta = max(0.0, float(n_containers))
    if str(_enum_value(business_type)).lower() == "export":
        return 0.3 * move_delta, 0.7 * move_delta
    return 0.7 * move_delta, 0.3 * move_delta


def _fallback_max_moves(workload_snapshot: Optional[Any]) -> float:
    if workload_snapshot is not None:
        for workload in getattr(workload_snapshot, "by_area_step", {}).values():
            max_moves = _as_float(getattr(workload, "max_moves", 0.0))
            if max_moves > 0.0:
                return max_moves
    return 2.0 * 30.0 * 4.0


def _item(score: float, formula: str, **details: Any) -> Dict[str, Any]:
    return {
        "score": round(_clamp(score, 0.0, 10.0), 2),
        "maxScore": 10.0,
        "formula": formula,
        "details": details,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-9:
        return 0.0
    return numerator / denominator


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _fmt(value: Any) -> str:
    number = _as_float(value)
    return f"{number:.2f}".rstrip("0").rstrip(".")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run yard planning and print the 50-point score.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--line-key", dest="line_key", type=int, help="Service line key.")
    target.add_argument("--vessel-key", dest="vessel_key", type=int, default=15623707, help="VesselVisit dbkey.")
    parser.add_argument("--type", dest="plan_type", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Print the raw score JSON.")
    args = parser.parse_args()

    from yardplan import run_plan

    kwargs: Dict[str, Any] = {"type": args.plan_type, "save_visualization": False}
    if args.line_key is not None:
        kwargs["line_keys"] = args.line_key
    else:
        kwargs["vessel_key"] = args.vessel_key
    result = run_plan(**kwargs)
    score = getattr(result, "metrics", {}).get("score") or score_yard_plan(result)
    if args.json:
        print(json.dumps(score, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_score_report(score))


if __name__ == "__main__":
    main()
