from __future__ import annotations

from typing import Iterable, List, Optional

from yardplan_core.models import AllocationGroup


SIMULTANEOUS_LOADING_CONFLICT_GROUP_IDS = "simultaneous_loading_conflict_group_ids"
SIMULTANEOUS_LOADING_PAIR_IDS = "simultaneous_loading_pair_ids"
SIMULTANEOUS_LOADING_SAFETY_GAP_BAYS = "simultaneous_loading_safety_gap_bays"


def group_root_id(group: AllocationGroup) -> str:
    return str(group.parent_group_id or group.group_id)


def _as_string_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def simultaneous_loading_conflict_group_ids(group: AllocationGroup) -> List[str]:
    return _as_string_list(
        (group.group_attributes or {}).get(SIMULTANEOUS_LOADING_CONFLICT_GROUP_IDS)
    )


def simultaneous_loading_pair_ids(group: AllocationGroup) -> List[str]:
    return _as_string_list(
        (group.group_attributes or {}).get(SIMULTANEOUS_LOADING_PAIR_IDS)
    )


def simultaneous_loading_safety_gap_bays(
    group: AllocationGroup,
    default: Optional[int] = None,
) -> Optional[int]:
    if SIMULTANEOUS_LOADING_SAFETY_GAP_BAYS not in (group.group_attributes or {}):
        return default
    try:
        return max(0, int(group.group_attributes[SIMULTANEOUS_LOADING_SAFETY_GAP_BAYS]))
    except (TypeError, ValueError):
        return default


def groups_have_simultaneous_loading_conflict(
    group_a: AllocationGroup,
    group_b: AllocationGroup,
) -> bool:
    root_a = group_root_id(group_a)
    root_b = group_root_id(group_b)
    if root_a == root_b:
        return False
    return (
        root_b in set(simultaneous_loading_conflict_group_ids(group_a))
        or root_a in set(simultaneous_loading_conflict_group_ids(group_b))
    )


def clear_simultaneous_loading_markers(group: AllocationGroup) -> None:
    attrs = group.group_attributes
    attrs.pop(SIMULTANEOUS_LOADING_CONFLICT_GROUP_IDS, None)
    attrs.pop(SIMULTANEOUS_LOADING_PAIR_IDS, None)
    attrs.pop(SIMULTANEOUS_LOADING_SAFETY_GAP_BAYS, None)


def add_simultaneous_loading_conflict(
    group: AllocationGroup,
    other_root_id: str,
    pair_id: str,
    safety_gap_bays: int,
) -> None:
    attrs = group.group_attributes
    conflicts = simultaneous_loading_conflict_group_ids(group)
    if other_root_id not in conflicts:
        conflicts.append(other_root_id)
    pair_ids = simultaneous_loading_pair_ids(group)
    if pair_id not in pair_ids:
        pair_ids.append(pair_id)
    attrs[SIMULTANEOUS_LOADING_CONFLICT_GROUP_IDS] = conflicts
    attrs[SIMULTANEOUS_LOADING_PAIR_IDS] = pair_ids
    attrs[SIMULTANEOUS_LOADING_SAFETY_GAP_BAYS] = int(safety_gap_bays)
