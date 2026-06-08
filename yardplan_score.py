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

    group_by_id = {getattr(group, "group_id", ""): group for group in groups}
    facts = _build_allocation_facts(allocations, assignments, group_by_id)
    root_demands = _root_demands(groups, facts)

    items = {
        "demand_satisfaction": _score_demand_satisfaction(root_demands, facts),
        "business_dispersion": _score_business_dispersion(root_demands, facts),
        "crane_workload_balance": _score_crane_workload_balance(
            facts,
            yard_areas=yard_areas,
            workload_snapshot=workload_snapshot,
        ),
        "bay_utilization": _score_bay_utilization(
            allocations,
            yard_areas=yard_areas,
        ),
        "area_peak_staggering": _score_area_peak_staggering(
            facts,
            yard_areas=yard_areas,
            workload_snapshot=workload_snapshot,
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
        },
    }


def format_score_report(score: Dict[str, Any], *, indent: str = "") -> str:
    labels = {
        "demand_satisfaction": "Demand satisfaction",
        "business_dispersion": "Business dispersion",
        "crane_workload_balance": "Crane workload balance",
        "bay_utilization": "Bay utilization",
        "area_peak_staggering": "Area peak staggering",
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


def _score_business_dispersion(
    root_demands: Dict[str, float],
    facts: Sequence[AllocationFact],
) -> Dict[str, Any]:
    if not root_demands:
        return _item(0.0, "No demand groups")

    by_root_area: DefaultDict[str, DefaultDict[str, float]] = defaultdict(lambda: defaultdict(float))
    for fact in facts:
        by_root_area[fact.root_group_id][fact.yard_area_id] += max(0.0, fact.planned_containers)

    total_demand = sum(root_demands.values())
    weighted = 0.0
    details: Dict[str, Any] = {}
    for root_id, demand in root_demands.items():
        area_loads = by_root_area.get(root_id, {})
        assigned = sum(area_loads.values())
        if assigned <= 0.0:
            group_score = 0.0
            effective_areas = 0.0
        else:
            shares = [load / assigned for load in area_loads.values() if load > 0.0]
            effective_areas = 1.0 / max(1e-9, sum(share * share for share in shares))
            target = _target_area_count(demand)
            tolerance = max(1.0, float(target))
            group_score = max(0.0, 1.0 - abs(effective_areas - target) / tolerance)
        weighted += (demand / total_demand) * group_score
        details[root_id] = {
            "assignedAreas": len([v for v in area_loads.values() if v > 0.0]),
            "effectiveAreas": round(effective_areas, 3),
            "targetAreas": _target_area_count(demand),
        }

    return _item(
        10.0 * weighted,
        "1 - abs(effective_area_count - target_area_count) / tolerance",
        groups=details,
    )


def _score_crane_workload_balance(
    facts: Sequence[AllocationFact],
    *,
    yard_areas: Sequence[Any],
    workload_snapshot: Optional[Any],
) -> Dict[str, Any]:
    series = _build_workload_series(facts, yard_areas, workload_snapshot)
    if not series:
        return _item(0.0, "No workload data")

    by_step: DefaultDict[int, List[Tuple[str, float, float]]] = defaultdict(list)
    for (area_id, step_id), values in series.items():
        by_step[step_id].append((area_id, values["total"], values["max_moves"]))

    weighted_jain = 0.0
    total_weight = 0.0
    overload = 0.0
    capacity = 0.0
    step_details: Dict[str, Any] = {}
    for step_id, rows in sorted(by_step.items()):
        loads = [max(0.0, row[1]) for row in rows]
        total_load = sum(loads)
        if total_load <= 0.0:
            continue
        sumsq = sum(load * load for load in loads)
        jain = 1.0 if sumsq <= 0.0 else (total_load * total_load) / (len(loads) * sumsq)
        weighted_jain += total_load * jain
        total_weight += total_load
        overload += sum(max(0.0, load - max_moves) for _area, load, max_moves in rows)
        capacity += sum(max(0.0, max_moves) for _area, _load, max_moves in rows)
        step_details[str(step_id)] = {
            "jain": round(jain, 4),
            "totalMoves": round(total_load, 3),
        }

    if total_weight <= 0.0:
        return _item(0.0, "No active workload steps")
    balance = weighted_jain / total_weight
    overload_ratio = _safe_div(overload, capacity)
    score = 10.0 * balance * max(0.0, 1.0 - overload_ratio)
    return _item(
        score,
        "weighted Jain fairness * overload penalty",
        weightedJain=round(balance, 4),
        overloadRatio=round(overload_ratio, 4),
        steps=step_details,
    )


def _score_bay_utilization(
    allocations: Sequence[Any],
    *,
    yard_areas: Sequence[Any],
    full_threshold: float = 0.85,
) -> Dict[str, Any]:
    bay_capacity: Dict[Tuple[str, int], float] = {}
    bay_base_occupied: Dict[Tuple[str, int], float] = {}
    for area in yard_areas:
        area_id = str(getattr(area, "area_id", ""))
        for bay in getattr(area, "bays", []) or []:
            key = (area_id, int(getattr(bay, "bay_number", 0)))
            bay_capacity[key] = max(1.0, _as_float(getattr(bay, "total_columns", 0.0)))
            bay_base_occupied[key] = max(0.0, _as_float(getattr(bay, "occupied_columns", 0.0)))

    planned_by_bay: DefaultDict[Tuple[str, int], float] = defaultdict(float)
    for allocation in allocations:
        area_id = str(getattr(allocation, "yard_area_id", ""))
        for bay_number, columns in _allocation_columns_by_bay(allocation).items():
            key = (area_id, bay_number)
            planned_by_bay[key] += columns
            if key not in bay_capacity:
                bay_capacity[key] = max(1.0, columns)
                bay_base_occupied[key] = 0.0

    total_planned = sum(planned_by_bay.values())
    if total_planned <= 0.0:
        return _item(0.0, "No placed bay columns")

    activated_capacity = 0.0
    activated_final_occupied = 0.0
    planned_in_full_bays = 0.0
    for key, planned in planned_by_bay.items():
        capacity = max(1.0, bay_capacity.get(key, planned))
        final_occupied = min(capacity, bay_base_occupied.get(key, 0.0) + planned)
        utilization = _safe_div(final_occupied, capacity)
        activated_capacity += capacity
        activated_final_occupied += final_occupied
        if utilization >= full_threshold:
            planned_in_full_bays += planned

    full_box_ratio = _safe_div(planned_in_full_bays, total_planned)
    used_bay_util = _safe_div(activated_final_occupied, activated_capacity)
    score = 10.0 * (0.6 * full_box_ratio + 0.4 * min(used_bay_util / full_threshold, 1.0))
    return _item(
        score,
        "0.6*planned_columns_in_full_bays + 0.4*activated_bay_utilization",
        fullThreshold=full_threshold,
        fullBoxRatio=round(full_box_ratio, 4),
        activatedBayUtilization=round(used_bay_util, 4),
        plannedColumns=round(total_planned, 3),
    )


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
        else:
            overlap = sum(
                min(row["inbound"] / inbound_total, row["outbound"] / outbound_total)
                for row in rows
            )
        weighted += weight * (1.0 - overlap)
        total_weight += weight
        area_details[area_id] = {
            "overlap": round(overlap, 4),
            "moves": round(weight, 3),
        }

    if total_weight <= 0.0:
        return _item(0.0, "No active area workload")
    score = 10.0 * weighted / total_weight
    return _item(
        score,
        "weighted average of 1 - sum_t min(in_share_t, out_share_t)",
        areas=area_details,
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
