from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


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
        "business_dispersion": _score_business_dispersion(root_demands, facts),
        "area_peak_staggering": _score_area_peak_staggering(
            facts,
            yard_areas=yard_areas,
            workload_snapshot=workload_snapshot,
        ),
        "bay_utilization_purity": _score_bay_utilization_purity(
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
        "bay_utilization_purity": "Bay utilization / purity",
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
    series = _build_workload_series(facts, yard_areas, workload_snapshot)
    if not series:
        return _item(0.0, "No workload data")

    by_area: DefaultDict[str, List[Dict[str, float]]] = defaultdict(list)
    for (area_id, _step_id), values in series.items():
        by_area[area_id].append(values)

    weighted = 0.0
    total_weight = 0.0
    area_details: Dict[str, Any] = {}
    for area_id, rows in sorted(by_area.items()):
        inbound_total = sum(row["inbound"] for row in rows)
        outbound_total = sum(row["outbound"] for row in rows)
        weight = inbound_total + outbound_total
        if weight <= 0.0:
            continue

        if inbound_total <= 0.0 or outbound_total <= 0.0:
            overlap = 0.0
            peak_gap = 0.0
            score = 10.0
        else:
            inbound_share = [row["inbound"] / inbound_total for row in rows]
            outbound_share = [row["outbound"] / outbound_total for row in rows]
            overlap = sum(min(i, o) for i, o in zip(inbound_share, outbound_share))
            inbound_peak = max(range(len(rows)), key=lambda idx: rows[idx]["inbound"])
            outbound_peak = max(range(len(rows)), key=lambda idx: rows[idx]["outbound"])
            peak_gap = abs(inbound_peak - outbound_peak)
            gap_bonus = min(1.0, peak_gap / 2.0)
            score = 10.0 * (
                0.7 * (1.0 - _clamp(overlap, 0.0, 1.0))
                + 0.3 * gap_bonus
            )

        weighted += weight * score
        total_weight += weight
        area_details[area_id] = {
            "overlap": round(overlap, 4),
            "peakGap": int(peak_gap),
            "moves": round(weight, 3),
            "score": round(score, 2),
        }

    if total_weight <= 0.0:
        return _item(0.0, "No active area workload")
    return _item(
        weighted / total_weight,
        "0.7*(1-overlap) + 0.3*min(peak_gap/2, 1)",
        areas=area_details,
    )


def _score_bay_utilization_purity(
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

    for allocation in allocations:
        group_id = str(getattr(allocation, "group_id", ""))
        area_id = str(getattr(allocation, "yard_area_id", ""))
        if not group_id or not area_id:
            continue
        group = group_by_id.get(group_id)
        size = _enum_value(getattr(group, "size", getattr(allocation, "size", None)))
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
    score = 10.0 * (0.6 * utilization_score + 0.4 * purity_score)

    return _item(
        score,
        "0.6*bay_utilization_score + 0.4*bay_purity_score",
        fullThreshold=full_threshold,
        fullRatio=round(full_ratio, 4),
        activatedUtilization=round(activated_util, 4),
        utilizationScore=round(utilization_score, 4),
        purityScore=round(purity_score, 4),
        plannedColumns=round(total_planned, 3),
        bays=bay_details,
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
) -> Dict[str, Any]:
    if not root_demands:
        return _item(0.0, "No demand groups")

    by_root_area: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    root_groups: Dict[str, Any] = {}
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
            details[root_id] = {
                "score": 0.0,
                "reason": "unassigned",
                "voyageId": voyage_id,
            }
            continue

        assigned_areas = len([v for v in area_loads.values() if v > 0.0])
        eqp_num = _group_eqp_num(group)
        if eqp_num <= 0:
            group_score = 0.0
            status = "missing crane count"
            target_areas = None
        else:
            target_areas = 2 * eqp_num
            if assigned_areas < target_areas - 1 or assigned_areas > target_areas + 2:
                group_score = 0.0
            else:
                group_score = max(0.0, 10.0 - 2.0 * abs(assigned_areas - target_areas))
            status = "target=2*crane_count, accept [-1,+2]"

        weighted += (demand / total_demand) * group_score
        details[root_id] = {
            "score": round(group_score, 2),
            "voyageId": voyage_id,
            "cranes": eqp_num,
            "targetAreas": target_areas,
            "assignedAreas": assigned_areas,
            "status": status,
        }

    return _item(
        10.0 * weighted,
        "10 - 2*abs(assigned_area_count - 2*crane_count), zero outside [-1,+2]",
        groups=details,
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


def _bay_numbers(bay_spec: Any) -> List[int]:
    if isinstance(bay_spec, (list, tuple)):
        return [int(value) for value in bay_spec]
    return [int(bay_spec)]


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
