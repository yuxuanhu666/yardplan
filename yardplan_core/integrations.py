from __future__ import annotations

import json
import importlib
import os
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from yardplan_core.models import (
    _DATA_DIR,
    AllocationGroup,
    Bay,
    BusinessType,
    Container,
    ContainerSize,
    ContainerType,
    LargeBayPair,
    MAX_TIERS_PER_COLUMN,
    PlanningResult,
    Vessel,
    WeightClass,
    YardArea,
    logger,
)


class YardSpaceAdapter:
    """
    将 YardSpace 的实时状态转换为规划引擎所需的 YardArea/Bay 对象。
    """

    @staticmethod
    def build_yard_areas(
        yard: Any,
        block_business_types: Dict[str, BusinessType],
        block_ids: Optional[List[str]] = None,
    ) -> Tuple[List[YardArea], Dict[str, list]]:
        blocks: Dict[str, dict] = defaultdict(dict)
        for (block_id, bay_idx, stack_idx), stack_info in yard.stacks.items():
            blocks[block_id][(bay_idx, stack_idx)] = stack_info

        slot_registry: Dict[str, list] = {}
        yard_areas: List[YardArea] = []
        target_blocks = block_ids if block_ids else sorted(blocks.keys())

        for block_id in target_blocks:
            if block_id not in blocks:
                continue

            business_type = block_business_types.get(block_id, BusinessType.IMPORT)
            stack_dict = blocks[block_id]
            bay_groups: Dict[int, list] = defaultdict(list)

            for (bay_idx, stack_idx), stack_info in stack_dict.items():
                bay_groups[bay_idx].append(((block_id, bay_idx, stack_idx), stack_info))

            sorted_bay_idxs = sorted(bay_groups.keys())
            first_bay_idx = sorted_bay_idxs[0]
            last_bay_idx = sorted_bay_idxs[-1]
            bays: List[Bay] = []
            bays_by_number: Dict[int, Bay] = {}
            large_bay_pairs: List[LargeBayPair] = []
            stage2_single_slots: List[Dict[str, int]] = []
            stage2_large_slots: List[Dict[str, Any]] = []
            existing_20ft_bays, existing_large_bays = YardSpaceAdapter._existing_bay_locks(
                yard,
                block_id,
                stack_dict,
                sorted_bay_idxs,
            )

            for bay_idx in sorted_bay_idxs:
                stacks_in_bay = bay_groups[bay_idx]
                total = len(stacks_in_bay)
                free = sum(
                    1
                    for _, stack_info in stacks_in_bay
                    if stack_info.get("next_placeable_tier") is not None
                )
                occupied = total - free
                bay_obj = Bay(
                    bay_id=f"{block_id}-{bay_idx}",
                    bay_number=bay_idx,
                    yard_area_id=block_id,
                    total_columns=total,
                    occupied_columns=occupied,
                    is_in_large_bay=False,
                )
                bays.append(bay_obj)
                bays_by_number[bay_idx] = bay_obj
                slot_registry[bay_obj.bay_id] = stacks_in_bay

                for (stack_key, stack_info) in stacks_in_bay:
                    bay_number = int(stack_key[1])
                    if bay_number in existing_large_bays:
                        continue
                    tiers = YardSpaceAdapter._remaining_20ft_tiers(stack_info)
                    if tiers:
                        stage2_single_slots.append(
                            {
                                "bay_number": bay_number,
                                "stack_index": int(stack_key[2]),
                                "tiers": tiers,
                            }
                        )

            seen_large_slots: Set[Tuple[Tuple[int, int], int]] = set()
            for pair_start, pair_end in zip(sorted_bay_idxs[::2], sorted_bay_idxs[1::2]):
                if pair_end != pair_start + 1:
                    logger.warning(
                        f"Block {block_id} 跳过非相邻 40ft 大贝对 ({pair_start}, {pair_end})"
                    )
                    continue

                pair_bays = (pair_start, pair_end)
                is_edge = pair_start == first_bay_idx or pair_end == last_bay_idx
                pair = LargeBayPair(
                    pair_id=f"{block_id}-40-{pair_start}_{pair_end}",
                    yard_area_id=block_id,
                    bay_a=bays_by_number[pair_start],
                    bay_b=bays_by_number[pair_end],
                    is_edge_pair=is_edge,
                )
                large_bay_pairs.append(pair)
                slot_registry[pair.pair_id] = [
                    *bay_groups.get(pair_start, []),
                    *bay_groups.get(pair_end, []),
                ]

                for bay_idx in pair_bays:
                    for (stack_key, stack_info) in bay_groups.get(bay_idx, []):
                        tiers = YardSpaceAdapter._remaining_large_tiers(yard, stack_info)
                        if not tiers:
                            continue
                        slot_40ft = YardSpaceAdapter._first_tier_value(
                            stack_info,
                            "slot_40ft",
                        )
                        actual_bays = YardSpaceAdapter._related_40ft_bays(
                            yard,
                            slot_40ft,
                            fallback_bays=pair_bays,
                        )
                        if tuple(sorted(actual_bays)) != pair_bays:
                            continue
                        if (
                            set(pair_bays) & existing_20ft_bays
                            or set(pair_bays) & existing_large_bays
                        ):
                            continue
                        stack_index = int(stack_key[2])
                        slot_key = (pair_bays, stack_index)
                        if slot_key in seen_large_slots:
                            continue
                        seen_large_slots.add(slot_key)
                        stage2_large_slots.append(
                            {
                                "pair_id": pair.pair_id,
                                "bay_numbers": pair_bays,
                                "display_bays": pair_bays,
                                "stack_index": stack_index,
                                "tiers": tiers,
                                "is_edge_pair": is_edge,
                            }
                        )

            area = YardArea(
                area_id=block_id,
                business_type=business_type,
                bays=bays,
                large_bay_pairs=large_bay_pairs,
                max_stack_height=MAX_TIERS_PER_COLUMN,
                center_coordinate=YardSpaceAdapter._block_center_coordinate(stack_dict),
            )
            area._stage2_single_slots = stage2_single_slots
            area._stage2_large_slots = stage2_large_slots
            area._stage2_existing_20ft_bays = existing_20ft_bays
            area._stage2_existing_large_bays = existing_large_bays
            yard_areas.append(area)

        logger.info(
            f"YardSpaceAdapter: 构建了 {len(yard_areas)} 个 YardArea "
            f"(20ft Bay: {sum(len(area.bays) for area in yard_areas)}, "
            f"40ft LargeBayPair: {sum(len(area.large_bay_pairs) for area in yard_areas)})"
        )
        return yard_areas, slot_registry

    @staticmethod
    def _existing_bay_locks(
        yard: Any,
        block_id: str,
        stack_dict: Dict[Tuple[int, int], Dict[str, Any]],
        bay_numbers: Optional[List[int]] = None,
    ) -> Tuple[Set[int], Set[int]]:
        existing_20ft_bays: Set[int] = set()
        existing_large_bays: Set[int] = set()
        block_bays = sorted(
            bay_numbers
            if bay_numbers is not None
            else {bay_idx for bay_idx, _stack_idx in stack_dict}
        )

        for (bay_idx, _stack_idx), stack_info in stack_dict.items():
            for tier_data in (stack_info.get("tiers") or {}).values():
                occupant = tier_data.get("occupant")
                if occupant:
                    size = occupant.get("size")
                    if size == "20ft":
                        existing_20ft_bays.add(int(bay_idx))
                    elif size in ("40ft", "45ft"):
                        existing_large_bays.update(
                            YardSpaceAdapter._canonical_40ft_pair_for_bay(
                                int(bay_idx),
                                block_bays,
                            )
                        )

                slot_40ft = tier_data.get("slot_40ft")
                if slot_40ft and slot_40ft in getattr(yard, "occupied_40ft", set()):
                    related_bays = YardSpaceAdapter._related_40ft_bays(
                        yard,
                        slot_40ft,
                        fallback_bays=YardSpaceAdapter._canonical_40ft_pair_for_bay(
                            int(bay_idx),
                            block_bays,
                        ),
                    )
                    existing_large_bays.update(related_bays)

        # A 40/45ft container may be registered by its even-bay master slot only.
        for slot_name in getattr(yard, "occupied_40ft", set()):
            slot_info = getattr(yard, "slots_40ft", {}).get(slot_name, {})
            if slot_info.get("blockId") != block_id:
                continue
            bay_idx = int(slot_info.get("bayIdx", 0))
            related_bays = YardSpaceAdapter._related_40ft_bays(
                yard,
                slot_name,
                fallback_bays=YardSpaceAdapter._canonical_40ft_pair_for_bay(
                    bay_idx,
                    block_bays,
                ),
            )
            existing_large_bays.update(related_bays)

        return existing_20ft_bays, existing_large_bays

    @staticmethod
    def _canonical_40ft_pair_for_bay(
        bay_idx: int,
        bay_numbers: List[int],
    ) -> Tuple[int, int]:
        """
        Map any physical bay to the non-overlapping 40ft pair sequence:
        (1,2), (3,4), (5,6), ... within a block's bay axis.
        """
        ordered = sorted(set(int(value) for value in bay_numbers))
        for left, right in zip(ordered[::2], ordered[1::2]):
            if bay_idx in (left, right):
                return (left, right)
        return (bay_idx, bay_idx + 1)

    @staticmethod
    def _is_empty_20ft_column(stack_info: Dict[str, Any]) -> bool:
        if stack_info.get("top_occupied_tier", 0) > 0:
            return False
        if stack_info.get("next_placeable_tier") is None:
            return False
        return bool(YardSpaceAdapter._first_tier_value(stack_info, "free_20ft"))

    @staticmethod
    def _is_empty_40ft_column(stack_info: Dict[str, Any]) -> bool:
        if stack_info.get("top_occupied_tier", 0) > 0:
            return False
        if stack_info.get("next_placeable_tier") is None:
            return False
        if not YardSpaceAdapter._first_tier_value(stack_info, "slot_40ft"):
            return False
        return bool(YardSpaceAdapter._first_tier_value(stack_info, "free_40ft"))

    @staticmethod
    def _remaining_20ft_tiers(stack_info: Dict[str, Any]) -> List[int]:
        top = int(stack_info.get("top_occupied_tier") or 0)
        max_tier = int(stack_info.get("max_tier") or MAX_TIERS_PER_COLUMN)
        tiers_data = stack_info.get("tiers") or {}
        tiers: List[int] = []
        for tier in range(top + 1, max_tier + 1):
            tier_data = tiers_data.get(tier, {})
            if not tier_data.get("free_20ft", False):
                break
            tiers.append(tier)
        return tiers

    @staticmethod
    def _remaining_large_tiers(
        yard: Any,
        stack_info: Dict[str, Any],
    ) -> List[int]:
        top = int(stack_info.get("top_occupied_tier") or 0)
        max_tier = int(stack_info.get("max_tier") or MAX_TIERS_PER_COLUMN)
        tiers_data = stack_info.get("tiers") or {}
        tiers: List[int] = []

        for tier in range(top + 1, max_tier + 1):
            tier_data = tiers_data.get(tier, {})
            slot_40ft = tier_data.get("slot_40ft")
            if not slot_40ft or not tier_data.get("free_40ft", False):
                break
            if not tiers and not YardSpaceAdapter._large_tier_supported(
                yard,
                slot_40ft,
                tier,
            ):
                break
            tiers.append(tier)
        return tiers

    @staticmethod
    def _large_tier_supported(yard: Any, slot_40ft: str, tier: int) -> bool:
        if tier <= 1:
            return True
        slot_info = getattr(yard, "slots_40ft", {}).get(slot_40ft, {})
        for related_name in slot_info.get("related_20ft", []):
            related = getattr(yard, "slots_20ft", {}).get(related_name)
            if not related:
                continue
            related_stack = yard.stacks.get(
                (
                    related["blockId"],
                    related["bayIdx"],
                    related["stackIdx"],
                )
            )
            if related_stack and int(related_stack.get("top_occupied_tier") or 0) < tier - 1:
                return False
        return True

    @staticmethod
    def _first_tier_value(stack_info: Dict[str, Any], key: str) -> Any:
        tiers = stack_info.get("tiers") or {}
        if not tiers:
            return None
        first_tier = min(tiers)
        return tiers.get(first_tier, {}).get(key)

    @staticmethod
    def _related_40ft_bays_for_group(
        yard: Any,
        stacks_in_bay: list,
        fallback_bays: Tuple[int, int],
    ) -> Tuple[int, int]:
        for _stack_key, stack_info in stacks_in_bay:
            slot_40ft = YardSpaceAdapter._first_tier_value(stack_info, "slot_40ft")
            related = YardSpaceAdapter._related_40ft_bays(
                yard,
                slot_40ft,
                fallback_bays=fallback_bays,
            )
            if related:
                return related
        return fallback_bays

    @staticmethod
    def _related_40ft_bays(
        yard: Any,
        slot_40ft: Optional[str],
        fallback_bays: Tuple[int, int],
    ) -> Tuple[int, int]:
        if not slot_40ft:
            return fallback_bays
        slot_info = getattr(yard, "slots_40ft", {}).get(slot_40ft, {})
        bay_numbers: List[int] = []
        for related_name in slot_info.get("related_20ft", []):
            related_info = getattr(yard, "slots_20ft", {}).get(related_name)
            if related_info is not None:
                bay_numbers.append(int(related_info["bayIdx"]))
        unique_bays = sorted(set(bay_numbers))
        if len(unique_bays) >= 2:
            return unique_bays[0], unique_bays[-1]
        return fallback_bays

    @staticmethod
    def _block_center_coordinate(
        stack_dict: Dict[Tuple[int, int], Dict[str, Any]],
    ) -> Optional[Tuple[float, float]]:
        points: List[Tuple[float, float]] = []
        for stack_info in stack_dict.values():
            for tier_data in (stack_info.get("tiers") or {}).values():
                coord = tier_data.get("coordinate") or {}
                if "x" not in coord or "y" not in coord:
                    continue
                try:
                    points.append((float(coord["x"]), float(coord["y"])))
                except (TypeError, ValueError):
                    continue
                break

        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0

    @staticmethod
    def apply_allocation(
        result: PlanningResult,
        yard: Any,
    ) -> Dict[str, str]:
        group_by_id: Dict[str, AllocationGroup] = {
            group.group_id: group for group in result.allocation_groups
        }
        assignment_map: Dict[str, str] = {}

        for allocation in result.bay_column_allocations:
            group = group_by_id.get(allocation.group_id)
            if group is None:
                continue

            block_id = allocation.yard_area_id
            is_large = allocation.size in (
                ContainerSize.SIZE_40,
                ContainerSize.SIZE_45,
            )
            size_int = {
                ContainerSize.SIZE_20: 1,
                ContainerSize.SIZE_40: 2,
                ContainerSize.SIZE_45: 3,
            }.get(allocation.size, 1)

            for container in group.containers:
                pool = (
                    yard.get_placeable_40ft(block_id=block_id)
                    if is_large
                    else yard.get_placeable_20ft(block_id=block_id)
                )
                if not pool:
                    logger.warning(
                        f"Block {block_id} 无可用槽位, 容器 {container.container_id} 未能分配"
                    )
                    break

                slot_name = pool[0]["fullSlotName"]
                yard.place_container(slot_name, container.container_id, size_int)
                assignment_map[container.container_id] = slot_name

        logger.info(
            f"YardSpaceAdapter.apply_allocation: 共分配 {len(assignment_map)} 个容器"
        )
        return assignment_map


class TOSLoader:
    """
    从 TOS 接口下载的 JSON 文件中加载规划所需数据。
    """

    _ISO_TYPE_MAP: Dict[str, ContainerType] = {
        "RF": ContainerType.REEFER,
        "OT": ContainerType.OPEN_TOP,
        "PL": ContainerType.FLAT_RACK,
        "FR": ContainerType.FLAT_RACK,
        "TK": ContainerType.TANK,
    }

    _SIZE_MAP: Dict[int, ContainerSize] = {
        1: ContainerSize.SIZE_20,
        2: ContainerSize.SIZE_40,
        3: ContainerSize.SIZE_45,
    }

    HEAVY_WEIGHT_THRESHOLD: float = 20000.0

    def __init__(
        self,
        vessel_visit_path: Optional[str] = None,
        bound_list_path: Optional[str] = None,
        berth_plan_path: Optional[str] = None,
        berth_data_path: Optional[str] = None,
        wq_path: Optional[str] = None,
        space_allocation_path: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.token = token
        self.vessel_visit_path = vessel_visit_path or os.path.join(
            _DATA_DIR,
            "getVesselVisit_new.json",
        )
        self.bound_list_path = bound_list_path or os.path.join(
            _DATA_DIR,
            "getBoundList _new.json",
        )

        self.berth_plan_path = berth_plan_path or os.path.join(
            _DATA_DIR,
            "berthplan.json",
        )
        self.berth_data_path = berth_data_path or os.path.join(
            _DATA_DIR,
            "berthdata.json",
        )

        self.wq_path = wq_path or os.path.join(_DATA_DIR, "217_WQ.json")
        self.space_allocation_path = space_allocation_path or os.path.join(
            _DATA_DIR,
            "getSpcaeAllocation(堆存计划范围).json",
        )
        self._json_cache: Dict[str, Any] = {}

    def _load_json_data(
        self,
        data_name: str,
        static_path: str,
        *,
        dynamic_func_name: Optional[str] = None,
        expected_type: Optional[type] = None,
        required_key: Optional[str] = None,
    ) -> Any:
        if data_name in self._json_cache:
            return self._json_cache[data_name]

        dynamic_reason = ""
        if dynamic_func_name and self.token:
            try:
                raw = self._call_url_receive(dynamic_func_name)
                raw = self._normalize_loaded_payload(
                    raw,
                    expected_type=expected_type,
                    required_key=required_key,
                )
                self._validate_loaded_payload(
                    data_name,
                    raw,
                    expected_type=expected_type,
                    required_key=required_key,
                )
                self._json_cache[data_name] = raw
                self._log_data_source(data_name, "dynamic", dynamic_func_name)
                return raw
            except Exception as exc:
                dynamic_reason = f"; dynamic failed: {exc}"
        elif dynamic_func_name:
            dynamic_reason = "; dynamic skipped: no token"

        try:
            with open(static_path, "r", encoding="utf-8") as file:
                raw = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to load static {data_name}: {static_path}") from exc

        raw = self._normalize_loaded_payload(
            raw,
            expected_type=expected_type,
            required_key=required_key,
        )
        self._validate_loaded_payload(
            data_name,
            raw,
            expected_type=expected_type,
            required_key=required_key,
        )
        self._json_cache[data_name] = raw
        detail = static_path if not dynamic_reason else f"{static_path}{dynamic_reason}"
        self._log_data_source(data_name, "static", detail)
        return raw

    def _call_url_receive(self, function_name: str) -> Any:
        try:
            url_receive = importlib.import_module("yardplan_core.URL_Receive")
        except ModuleNotFoundError as exc:
            if exc.name != "yardplan_core.URL_Receive":
                raise
            url_receive = importlib.import_module("URL_Receive")

        receiver: Callable[[Any], Any] = getattr(url_receive, function_name)
        return receiver(self.token)

    @staticmethod
    def _normalize_loaded_payload(
        raw: Any,
        *,
        expected_type: Optional[type],
        required_key: Optional[str],
    ) -> Any:
        if expected_type is list and isinstance(raw, dict):
            data = raw.get("data")
            if isinstance(data, list):
                return data
        if expected_type is dict and required_key and isinstance(raw, dict):
            if required_key not in raw:
                data = raw.get("data")
                if isinstance(data, dict) and required_key in data:
                    return data
        return raw

    @staticmethod
    def _validate_loaded_payload(
        data_name: str,
        raw: Any,
        *,
        expected_type: Optional[type],
        required_key: Optional[str],
    ) -> None:
        if expected_type is not None and not isinstance(raw, expected_type):
            raise ValueError(
                f"{data_name} expected {expected_type.__name__}, got {type(raw).__name__}"
            )
        if required_key and isinstance(raw, dict) and required_key not in raw:
            raise ValueError(f"{data_name} missing required key {required_key!r}")

    @staticmethod
    def _log_data_source(data_name: str, source: str, detail: str) -> None:
        message = f"[DATA] {data_name}: {source} ({detail})"
        print(message)
        logger.info(message)

    def _load_vessel_visit_json(self) -> List[dict]:
        return self._load_json_data(
            "vessel_visit",
            self.vessel_visit_path,
            dynamic_func_name="ShipVisitList_receive",
            expected_type=list,
        )

    def _load_bound_list_json(self) -> Dict[str, Any]:
        return self._load_json_data(
            "bound_list",
            self.bound_list_path,
            dynamic_func_name="BoundList_receive",
            expected_type=dict,
            required_key="Outbound",
        )

    def _load_wq_json(self) -> Dict[str, Any]:
        return self._load_json_data(
            "wq",
            self.wq_path,
            dynamic_func_name="WqByVesselVisit_receive",
            expected_type=dict,
        )

    def _load_berth_plan_json(self) -> Dict[str, Any]:
        return self._load_json_data(
            "berth_plan",
            self.berth_plan_path,
            dynamic_func_name="berthplan_receive",
            expected_type=dict,
        )

    def _load_berth_data_json(self) -> Dict[str, Any]:
        return self._load_json_data(
            "berth_data",
            self.berth_data_path,
            expected_type=dict,
            required_key="berthMap",
        )

    def _load_space_allocation_json(self) -> Dict[str, Any]:
        return self._load_json_data(
            "space_allocation",
            self.space_allocation_path,
            dynamic_func_name="SpaceAllocation_receive",
            expected_type=dict,
            required_key="groupMap",
        )

    def _load_berth_plans_by_visit_key(self) -> Dict[int, Dict[str, Any]]:
        try:
            raw = self._load_berth_plan_json()
        except ValueError as exc:
            logger.warning("TOSLoader: failed to load berth plans: %s", exc)
            return {}

        plans = raw.get("data", {}).get("listBerthPlan", []) if isinstance(raw, dict) else []
        by_visit_key: Dict[int, Dict[str, Any]] = {}
        for plan in plans:
            visit_key = self._coerce_line_key(plan.get("vesselVisitKey"))
            if visit_key is None:
                continue
            current = by_visit_key.get(visit_key)
            if current is None or self._berth_plan_sort_key(plan) < self._berth_plan_sort_key(current):
                by_visit_key[visit_key] = plan
        return by_visit_key

    def _load_berth_centers(self) -> Dict[int, Tuple[float, float]]:
        try:
            raw = self._load_berth_data_json()
        except ValueError as exc:
            logger.warning("TOSLoader: failed to load berth data: %s", exc)
            return {}

        centers: Dict[int, Tuple[float, float]] = {}
        berth_map = raw.get("berthMap", {}) if isinstance(raw, dict) else {}
        for raw_key, berth in berth_map.items():
            berth_key = self._coerce_line_key(berth.get("berthKey", raw_key))
            start = berth.get("startCoordinate") or {}
            end = berth.get("endCoordinate") or {}
            if berth_key is None:
                continue
            try:
                centers[berth_key] = (
                    (float(start["x"]) + float(end["x"])) / 2.0,
                    (float(start["y"]) + float(end["y"])) / 2.0,
                )
            except (KeyError, TypeError, ValueError):
                continue
        return centers

    @staticmethod
    def _berth_plan_sort_key(plan: Dict[str, Any]) -> Tuple[int, str, int]:
        seq = plan.get("seq")
        try:
            seq_value = int(seq) if seq is not None else 999999
        except (TypeError, ValueError):
            seq_value = 999999
        return seq_value, str(plan.get("eta") or ""), int(plan.get("dbkey") or 0)

    def load_departed_loading_container_ids(
        self,
        *,
        target_vessel_key: int,
        target_eta: datetime,
        plan_start_time: Optional[datetime] = None,
        plan_end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        target_key = self._coerce_line_key(target_vessel_key)
        empty_result = {
            "candidate_vessels": [],
            "container_ids": [],
            "containers_by_vessel": {},
        }
        if target_key is None:
            return empty_result

        try:
            visits = self._load_vessel_visit_json()
        except ValueError as exc:
            logger.warning("TOSLoader: failed to load vessel visits for yard cleanup: %s", exc)
            return empty_result

        candidates: Dict[int, Dict[str, Any]] = {}
        for item in visits:
            visit_key = self._coerce_line_key(item.get("dbkey"))
            if visit_key is None or visit_key == target_key:
                continue

            etd = self._parse_dt(item.get("etd"))
            if etd is None or etd >= target_eta:
                continue
            if plan_start_time is not None and etd < plan_start_time:
                continue
            if plan_end_time is not None and etd >= plan_end_time:
                continue

            candidates[visit_key] = {
                "vesselVisitKey": visit_key,
                "vesselVisitId": item.get("vesselVisitId") or "",
                "etd": etd,
            }

        if not candidates:
            return empty_result

        try:
            raw = self._load_wq_json()
        except ValueError as exc:
            logger.warning("TOSLoader: failed to load WQ data for yard cleanup: %s", exc)
            return {
                "candidate_vessels": list(candidates.values()),
                "container_ids": [],
                "containers_by_vessel": {},
            }

        wq_list = raw.get("data", {}).get("wqList", []) if isinstance(raw, dict) else []
        container_ids: List[str] = []
        seen: Set[str] = set()
        containers_by_vessel: Dict[int, List[str]] = defaultdict(list)

        for wq in wq_list:
            visit_key = self._coerce_line_key(wq.get("vesselVisitKey"))
            if visit_key not in candidates:
                continue
            if not self._is_loading_wq(wq):
                continue
            for container_id in wq.get("wiContrIdList") or []:
                if not container_id or container_id in seen:
                    continue
                seen.add(container_id)
                container_ids.append(container_id)
                containers_by_vessel[visit_key].append(container_id)

        candidate_vessels = list(candidates.values())
        for vessel in candidate_vessels:
            vessel["loadingContainerCount"] = len(
                containers_by_vessel.get(vessel["vesselVisitKey"], [])
            )

        return {
            "candidate_vessels": candidate_vessels,
            "container_ids": container_ids,
            "containers_by_vessel": dict(containers_by_vessel),
        }

    @staticmethod
    def _is_loading_wq(wq: Dict[str, Any]) -> bool:
        qtype = wq.get("qtype")
        if qtype == 1 or str(qtype).strip() == "1":
            return True
        wq_id = str(wq.get("wqId") or wq.get("name") or "").upper()
        return "LOAD" in wq_id

    def load_loading_wq_containers_for_window(
        self,
        *,
        target_vessel_key: int,
        window_start: datetime,
        window_end: datetime,
    ) -> Dict[str, Any]:
        target_key = self._coerce_line_key(target_vessel_key)
        empty_result = {
            "candidate_vessels": [],
            "containers_by_vessel": {},
        }
        if target_key is None or window_end <= window_start:
            return empty_result

        try:
            visits = self._load_vessel_visit_json()
        except ValueError as exc:
            logger.warning("TOSLoader: failed to load vessel visits for yard busy profile: %s", exc)
            return empty_result

        candidates: Dict[int, Dict[str, Any]] = {}
        for item in visits:
            visit_key = self._coerce_line_key(item.get("dbkey"))
            if visit_key is None or visit_key == target_key:
                continue
            eta = self._parse_dt(item.get("eta"))
            etd = self._parse_dt(item.get("etd"))
            if etd is None:
                continue
            if etd < window_start or etd >= window_end:
                continue
            candidates[visit_key] = {
                "vesselVisitKey": visit_key,
                "vesselVisitId": item.get("vesselVisitId") or "",
                "eta": eta,
                "etd": etd,
            }

        if not candidates:
            return empty_result

        try:
            raw = self._load_wq_json()
        except ValueError as exc:
            logger.warning("TOSLoader: failed to load WQ data for yard busy profile: %s", exc)
            return {
                "candidate_vessels": list(candidates.values()),
                "containers_by_vessel": {},
            }

        wq_list = raw.get("data", {}).get("wqList", []) if isinstance(raw, dict) else []
        containers_by_vessel: Dict[int, List[str]] = defaultdict(list)
        seen_by_vessel: Dict[int, Set[str]] = defaultdict(set)

        for wq in wq_list:
            visit_key = self._coerce_line_key(wq.get("vesselVisitKey"))
            if visit_key not in candidates:
                continue
            if not self._is_loading_wq(wq):
                continue
            seen = seen_by_vessel[visit_key]
            for container_id in wq.get("wiContrIdList") or []:
                if not container_id or container_id in seen:
                    continue
                seen.add(container_id)
                containers_by_vessel[visit_key].append(container_id)

        candidate_vessels = list(candidates.values())
        for vessel in candidate_vessels:
            vessel["loadingContainerCount"] = len(
                containers_by_vessel.get(vessel["vesselVisitKey"], [])
            )

        return {
            "candidate_vessels": candidate_vessels,
            "containers_by_vessel": dict(containers_by_vessel),
        }

    def load_vessels(
        self,
        line_keys: Optional[int] = None,
        vessel_key: Optional[int] = None,
        plan_start_time: Optional[datetime] = None,
        plan_end_time: Optional[datetime] = None,
    ) -> Dict[str, Vessel]:
        raw = self._load_vessel_visit_json()

        berth_plans = self._load_berth_plans_by_visit_key()
        berth_centers = self._load_berth_centers()

        target_set = self._coerce_line_key_set(line_keys)
        target_visit_dbkeys = self._coerce_line_key_set(vessel_key)
        vessels: Dict[str, Vessel] = {}
        matched_line_keys: Set[int] = set()
        matched_visit_dbkeys: Set[int] = set()
        skipped_by_plan_window = 0

        for item in raw:
            vessel_visit_id = item.get("vesselVisitId", "")
            line_key = self._coerce_line_key(item.get("lineKey"))
            item_visit_dbkey = self._coerce_line_key(item.get("dbkey"))
            if target_visit_dbkeys:
                if item_visit_dbkey is None or item_visit_dbkey not in target_visit_dbkeys:
                    continue
                matched_visit_dbkeys.add(item_visit_dbkey)
            else:
                if line_key not in target_set:
                    continue
                matched_line_keys.add(line_key)

            eta = self._parse_dt(item.get("eta"))
            etd = self._parse_dt(item.get("etd"))
            if plan_start_time is not None and plan_end_time is not None:
                if not self._visit_overlaps_plan_window(
                    eta, etd, plan_start_time, plan_end_time
                ):
                    skipped_by_plan_window += 1
                    continue

            vessel_info = item.get("vesselInfo") or {}
            vessel_id_str = vessel_info.get("id") or vessel_visit_id
            vessel_name = vessel_info.get("name") or ""
            berth_plan = berth_plans.get(item_visit_dbkey or -1, {})
            berth_key = self._coerce_line_key(berth_plan.get("berthKey"))
            berth_coordinate = berth_centers.get(berth_key) if berth_key is not None else None
            eqp_num = self._coerce_line_key(
                item.get("eqpNum") if item.get("eqpNum") is not None else item.get("EQP_NUM")
            )

            vessels[vessel_visit_id] = Vessel(
                vessel_id=vessel_id_str,
                vessel_name=vessel_name,
                voyage_id=vessel_visit_id,
                eta=eta or datetime.now(),
                etd=etd or (datetime.now() + timedelta(hours=48)),
                berth_id=str(berth_key or ""),
                berth_key=berth_key,
                berth_coordinate=berth_coordinate,
                eqp_num=eqp_num,
            )

        missing = target_set - matched_line_keys
        if missing:
            logger.warning(f"TOSLoader: VesselVisit 中未找到航线号: {sorted(missing)}")
        missing_visit_dbkeys = target_visit_dbkeys - matched_visit_dbkeys
        if missing_visit_dbkeys:
            logger.warning(
                f"TOSLoader: VesselVisit did not match dbkeys {sorted(missing_visit_dbkeys)}"
            )
        target_description = (
            f"visitDbkeys: {sorted(target_visit_dbkeys)}"
            if target_visit_dbkeys
            else f"lineKeys: {sorted(target_set)}"
        )
        if plan_start_time is not None and plan_end_time is not None:
            logger.info(
                f"TOSLoader: 规划范围 {plan_start_time} → {plan_end_time}，"
                f"加载船舶 {len(vessels)} 艘"
                f" ({target_description}，范围外跳过 {skipped_by_plan_window} 艘)"
            )
        else:
            logger.info(
                f"TOSLoader: 加载船舶 {len(vessels)} 艘"
                f" ({target_description})"
            )
        return vessels

    def load_discharge_containers(
        self,
        line_keys: Optional[int],
        vessels: Dict[str, Vessel],
        vessel_key: Optional[int] = None,
    ) -> List[Container]:
        return self._load_bound_containers(
            line_keys=line_keys,
            vessel_key=vessel_key,
            vessels=vessels,
            section_name="Inbound",
            business_type=BusinessType.IMPORT,
            expected_bound_type=1,
            expected_visit_type=1,
            description="卸船箱",
        )

    def load_loading_containers(
        self,
        line_keys: Optional[int],
        vessels: Dict[str, Vessel],
        vessel_key: Optional[int] = None,
    ) -> List[Container]:
        # 出口箱：217getBoundList 的 Outbound； boundType =2 时为出口箱
        return self._load_bound_containers(
            line_keys=line_keys,
            vessel_key=vessel_key,
            vessels=vessels,
            section_name="Outbound",
            business_type=BusinessType.EXPORT,
            expected_bound_type=2,
            expected_visit_type=1,
            description="装船箱",
        )

    def _load_bound_containers(
        self,
        line_keys: Optional[int],
        vessels: Dict[str, Vessel],
        vessel_key: Optional[int] = None,
        *,
        section_name: str,
        business_type: BusinessType,
        expected_bound_type: int,
        expected_visit_type: int,
        description: str,
    ) -> List[Container]:
        raw = self._load_bound_list_json()

        entries: List[dict] = raw.get(section_name, [])
        target_set = self._coerce_line_key_set(line_keys)
        target_visit_dbkeys = self._coerce_line_key_set(vessel_key)
        containers: List[Container] = []

        for item in entries:
            container_raw = item.get("container")
            if not container_raw:
                continue

            service_line_key = self._coerce_line_key(
                container_raw.get("serviceLineKey")
            )
            if not target_visit_dbkeys and service_line_key not in target_set:
                continue

            dto = item.get("boundListDTO") or {}
            visit_id = dto.get("visitId") or ""
            if dto.get("boundType") != expected_bound_type:
                continue
            if dto.get("visitType") != expected_visit_type:
                continue
            if visit_id not in vessels:
                continue

            vessel = vessels[visit_id]
            iso_type = container_raw.get("containerISO") or ""
            is_gauge = any(
                self._coerce_bool(container_raw.get(key))
                for key in (
                    "oog",
                    "overLongBack",
                    "overLongFront",
                    "overWidthLeft",
                    "overWidthRight",
                )
            )

            containers.append(
                Container(
                    container_id=container_raw.get("containerId")
                    or dto.get("contrId", ""),
                    size=self._parse_size(container_raw.get("containerSize", 1)),
                    container_type=self._parse_container_type(
                        iso_type,
                        container_raw.get("powerRequired", 0),
                    ),
                    weight_class=self._parse_weight_class(
                        container_raw.get("weight"),
                        container_raw.get("freightKind"),
                    ),
                    business_type=business_type,
                    voyage_id=visit_id,
                    line_key=service_line_key,
                    vessel_id=vessel.vessel_id if vessel else visit_id,
                    eta=vessel.eta if vessel else datetime.now(),
                    etd=vessel.etd if vessel else None,
                    destination_port=container_raw.get("pod"),
                    receiving_start=self._parse_dt(dto.get("aptDate"))
                    or self._parse_dt(container_raw.get("putTime")),
                    iso_type=iso_type,
                    category=container_raw.get("category"),
                    pod=container_raw.get("pod"),
                    cattier_kind=container_raw.get("cattierKind"),
                    trade_code=container_raw.get("tradeCode"),
                    service_line_code=container_raw.get("serviceLineCode"),
                    freight_kind=container_raw.get("freightKind"),
                    owner_company=container_raw.get("ownerCompany"),
                    line_company=container_raw.get("lineCompany"),
                    truck_company=container_raw.get("truckCompany"),
                    belonger_company=container_raw.get("gradesCompany"),
                    work_type=container_raw.get("workType"),
                    bol=container_raw.get("bol"),
                    damage_code=container_raw.get("damageType"),
                    raw_weight=container_raw.get("weight"),
                    is_reefer=self._coerce_bool(container_raw.get("powerRequired"))
                    or "RF" in iso_type.upper(),
                    is_hazardous=self._coerce_bool(container_raw.get("bHazardous")),
                    is_damage=self._coerce_bool(container_raw.get("damage"))
                    or bool(container_raw.get("damageType")),
                    is_high=self._coerce_bool(container_raw.get("overHeight")),
                    is_gauge=is_gauge,
                    is_dirty=self._coerce_bool(container_raw.get("dirty")),
                )
            )

        target_description = (
            f"visitDbkeys: {sorted(target_visit_dbkeys)}"
            if target_visit_dbkeys
            else f"lineKeys: {sorted(target_set)}"
        )
        logger.info(
            f"TOSLoader: 加载{description} {len(containers)} 个"
            f" ({target_description})"
        )
        if target_visit_dbkeys:
            logger.info(
                f"TOSLoader: loaded {description} {len(containers)} containers "
                f"(visitDbkeys: {sorted(target_visit_dbkeys)})"
            )
        return containers

    def build_type2_space_allocation_plan(
        self,
        *,
        containers: List[Container],
        yard: Any,
    ) -> Dict[str, Any]:
        raw_plan = self._load_space_allocation_plan()
        group_map = raw_plan.get("groupMap", {})
        if not isinstance(group_map, dict):
            raise ValueError("getSpcaeAllocation groupMap must be an object")

        ordered_groups = self._ordered_space_allocation_groups(group_map)
        containers_by_group: Dict[str, List[Container]] = defaultdict(list)
        unmatched_container_ids: List[str] = []

        for container in containers:
            group_key = self._match_space_allocation_group(container, ordered_groups)
            if group_key is None:
                unmatched_container_ids.append(container.container_id)
                continue
            containers_by_group[group_key].append(container)

        planned_ranges = self._all_active_space_ranges(group_map)
        updated_group_map: Dict[str, Dict[str, Any]] = {}
        summaries: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []

        for group_key, members in containers_by_group.items():
            original_group = group_map.get(group_key)
            if not isinstance(original_group, dict):
                continue

            original_ranges = self._normalize_range_list(original_group.get("rangeList"))
            demand_by_size = self._container_demand_by_size(members)
            existing_capacity_by_size = {
                size_int: self._range_capacity_for_size(yard, original_ranges, size_int)
                for size_int in (1, 2, 3)
            }

            deficit_by_size = {
                size_int: max(
                    0,
                    demand_by_size.get(size_int, 0)
                    - existing_capacity_by_size.get(size_int, 0),
                )
                for size_int in (1, 2, 3)
            }
            new_ranges: List[Dict[str, Any]] = []
            added_capacity_by_size: Dict[int, int] = {}
            next_range_seq = self._next_range_seq(original_ranges)

            for size_int in (1, 2, 3):
                deficit = deficit_by_size.get(size_int, 0)
                if deficit <= 0:
                    added_capacity_by_size[size_int] = 0
                    continue

                selected_ranges, added_capacity, next_range_seq = self._new_ranges_for_deficit(
                    yard=yard,
                    size_int=size_int,
                    deficit=deficit,
                    excluded_ranges=planned_ranges + new_ranges,
                    template_ranges=original_ranges,
                    next_range_seq=next_range_seq,
                )
                new_ranges.extend(selected_ranges)
                added_capacity_by_size[size_int] = added_capacity
                if added_capacity < deficit:
                    warnings.append(
                        f"groupKey={group_key} size={size_int} deficit={deficit} "
                        f"addedCapacity={added_capacity}"
                    )

            if new_ranges:
                updated_group = deepcopy(original_group)
                updated_group["rangeList"] = deepcopy(original_ranges) + new_ranges
                updated_group_map[group_key] = updated_group
                planned_ranges.extend(new_ranges)

            summaries[group_key] = {
                "groupName": original_group.get("groupName"),
                "containerCount": len(members),
                "demandBySize": demand_by_size,
                "existingCapacityBySize": existing_capacity_by_size,
                "deficitBySize": deficit_by_size,
                "addedCapacityBySize": added_capacity_by_size,
                "newRangeCount": len(new_ranges),
            }

        logger.info(
            "TOSLoader: type=2 matched %s/%s containers into %s groups; "
            "%s groups need new ranges",
            sum(len(items) for items in containers_by_group.values()),
            len(containers),
            len(containers_by_group),
            len(updated_group_map),
        )
        if unmatched_container_ids:
            logger.warning(
                "TOSLoader: type=2 unmatched containers: %s",
                unmatched_container_ids[:20],
            )
        if warnings:
            logger.warning("TOSLoader: type=2 range capacity warnings: %s", warnings[:20])

        api_data = self._space_group_map_to_range_plan_data(updated_group_map)
        return {
            "data": api_data,
            "groupMap": updated_group_map,
            "summary": summaries,
            "matchedContainerCount": sum(
                len(items) for items in containers_by_group.values()
            ),
            "unmatchedContainerIds": unmatched_container_ids,
            "warnings": warnings,
        }

    def _space_group_map_to_range_plan_data(
        self,
        group_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        for group_key, group in sorted(
            group_map.items(),
            key=lambda item: self._coerce_priority(item[0], default=999999999),
        ):
            group_key_int = self._coerce_line_key(group.get("groupKey"))
            if group_key_int is None:
                group_key_int = self._coerce_line_key(group_key)
            range_list = [
                self._space_range_item_for_api(range_item)
                for range_item in self._normalize_range_list(group.get("rangeList"))
            ]
            if not range_list:
                continue
            data.append(
                {
                    "groupKey": group_key_int,
                    "groupId": group_key_int,
                    "rangeList": range_list,
                    "filter": self._space_filter_for_api(group.get("filterAll") or {}),
                }
            )
        return data

    @staticmethod
    def _space_range_item_for_api(range_item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "blockId": str(range_item.get("blockId") or ""),
            "startBayIndex": int(range_item.get("startBayIndex") or 0),
            "endBayIndex": int(range_item.get("endBayIndex") or 0),
            "startStackIndex": int(range_item.get("startStackIndex") or 0),
            "endStackIndex": int(range_item.get("endStackIndex") or 0),
            "startTierIndex": int(range_item.get("startTierIndex") or 1),
            "endTierIndex": int(range_item.get("endTierIndex") or MAX_TIERS_PER_COLUMN),
        }

    def _space_filter_for_api(self, filter_all: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "filterName": str(filter_all.get("filterName") or ""),
            "isoType": self._space_filter_list(filter_all.get("isoType"), str),
            "category": self._space_filter_list(filter_all.get("category"), int),
            "pod": self._space_filter_list(filter_all.get("pod"), str),
            "cattierKind": self._space_filter_list(filter_all.get("cattierKind"), str),
            "tradeCode": self._space_filter_list(filter_all.get("tradeCode"), str),
            "freightKind": self._space_filter_list(filter_all.get("freightKind"), int),
            "bReefer": self._space_filter_bool(filter_all.get("bReefer")),
            "bHazardous": self._space_filter_bool(filter_all.get("bHazardous")),
            "bDamage": self._space_filter_bool(filter_all.get("bDamage")),
            "bHigh": self._space_filter_bool(filter_all.get("bHigh")),
            "bGauge": self._space_filter_bool(filter_all.get("bGauge")),
            "ownerCompany": self._space_filter_list(filter_all.get("ownerCompany"), str),
            "lineCompany": self._space_filter_list(filter_all.get("lineCompany"), str),
            "truckCompany": self._space_filter_list(filter_all.get("truckCompany"), str),
            "belongerCompany": self._space_filter_list(
                filter_all.get("belongerCompany"),
                str,
            ),
            "bDirty": self._space_filter_bool(filter_all.get("bDirty")),
            "weightClass": self._space_optional_int(filter_all.get("weightClass")),
            "weightMin": filter_all.get("weightMin"),
            "weightMax": filter_all.get("weightMax"),
            "workType": self._space_optional_int(filter_all.get("workType")),
            "bol": self._space_filter_list(filter_all.get("bol"), str),
            "damageCode": self._space_filter_list(filter_all.get("damageCode"), str),
        }

    @classmethod
    def _space_filter_list(cls, value: Any, converter: Any) -> List[Any]:
        if cls._wildcard(value):
            return []
        values = value if isinstance(value, (list, tuple, set)) else [value]
        result: List[Any] = []
        for item in values:
            if cls._wildcard(item):
                continue
            try:
                result.append(converter(item))
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _space_filter_bool(value: Any) -> bool:
        if value is None or value == "%":
            return False
        return TOSLoader._coerce_bool(value)

    @staticmethod
    def _space_optional_int(value: Any) -> Optional[int]:
        if value is None or value == "%":
            return None
        return TOSLoader._coerce_line_key(value)

    def _load_space_allocation_plan(self) -> Dict[str, Any]:
        try:
            raw = self._load_space_allocation_json()
        except ValueError as exc:
            raise ValueError(
                f"failed to load getSpcaeAllocation file: {self.space_allocation_path}"
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError("getSpcaeAllocation root must be an object")
        return raw

    def _ordered_space_allocation_groups(
        self,
        group_map: Dict[str, Any],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        groups = [
            (str(group_key), group)
            for group_key, group in group_map.items()
            if isinstance(group, dict)
        ]
        groups.sort(
            key=lambda item: (
                self._coerce_priority(item[1].get("groupPriority"), default=999999),
                -self._space_group_specificity(item[1]),
                self._coerce_priority(item[1].get("groupKey"), default=999999999),
                item[0],
            )
        )
        return groups

    def _match_space_allocation_group(
        self,
        container: Container,
        ordered_groups: List[Tuple[str, Dict[str, Any]]],
    ) -> Optional[str]:
        for group_key, group in ordered_groups:
            if self._space_allocation_group_matches(container, group):
                return group_key
        return None

    def _space_allocation_group_matches(
        self,
        container: Container,
        group: Dict[str, Any],
    ) -> bool:
        if not self._constraint_matches(group.get("category"), container.category, {-1}):
            return False
        if not self._constraint_matches(group.get("pod"), container.pod):
            return False
        if not self._vessel_constraint_matches(group.get("vesselId"), container):
            return False
        if not self._carrier_kind_matches(
            group.get("carrierKind"),
            group.get("carrierKindType"),
            container,
        ):
            return False

        filter_all = group.get("filterAll") or {}
        if not isinstance(filter_all, dict):
            return True

        field_map = {
            "isoType": "iso_type",
            "category": "category",
            "pod": "pod",
            "cattierKind": "cattier_kind",
            "tradeCode": "trade_code",
            "freightKind": "freight_kind",
            "bReefer": "is_reefer",
            "bHazardous": "is_hazardous",
            "bDamage": "is_damage",
            "bHigh": "is_high",
            "bGauge": "is_gauge",
            "ownerCompany": "owner_company",
            "lineCompany": "line_company",
            "truckCompany": "truck_company",
            "belongerCompany": "belonger_company",
            "bDirty": "is_dirty",
            "workType": "work_type",
            "bol": "bol",
            "damageCode": "damage_code",
        }
        for filter_key, container_attr in field_map.items():
            if not self._constraint_matches(
                filter_all.get(filter_key),
                getattr(container, container_attr, None),
            ):
                return False

        weight_min = filter_all.get("weightMin")
        weight_max = filter_all.get("weightMax")
        if weight_min is not None or weight_max is not None:
            weight = self._coerce_float(container.raw_weight)
            if weight is None:
                return False
            min_value = self._coerce_float(weight_min)
            max_value = self._coerce_float(weight_max)
            if min_value is not None and weight < min_value:
                return False
            if max_value is not None and weight > max_value:
                return False

        weight_class = filter_all.get("weightClass")
        if not self._wildcard(weight_class):
            actual = container.weight_class.value if container.weight_class else None
            if not self._constraint_matches(weight_class, actual):
                return False

        return True

    def _carrier_kind_matches(
        self,
        carrier_kind: Any,
        carrier_kind_type: Any,
        container: Container,
    ) -> bool:
        if self._wildcard(carrier_kind):
            return True

        kind_type = self._coerce_line_key(carrier_kind_type)
        if kind_type == 1:
            candidates = [container.service_line_code, container.line_key]
        elif kind_type == 2:
            candidates = [container.line_company]
        elif kind_type == 3:
            candidates = [container.voyage_id, container.vessel_id]
        else:
            candidates = [
                container.service_line_code,
                container.line_key,
                container.line_company,
                container.voyage_id,
                container.vessel_id,
            ]
        return any(self._constraint_matches(carrier_kind, candidate) for candidate in candidates)

    def _vessel_constraint_matches(
        self,
        vessel_constraint: Any,
        container: Container,
    ) -> bool:
        if self._wildcard(vessel_constraint):
            return True
        return any(
            self._constraint_matches(vessel_constraint, candidate)
            for candidate in (container.vessel_id, container.voyage_id)
        )

    @classmethod
    def _constraint_matches(
        cls,
        expected: Any,
        actual: Any,
        extra_wildcards: Optional[Set[Any]] = None,
    ) -> bool:
        if cls._wildcard(expected):
            return True
        if extra_wildcards and expected in extra_wildcards:
            return True
        if isinstance(expected, (list, tuple, set)):
            if not expected:
                return True
            return any(cls._constraint_matches(item, actual, extra_wildcards) for item in expected)
        if actual is None:
            return False
        return cls._values_equal(expected, actual)

    @staticmethod
    def _wildcard(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() in ("", "%")
        if isinstance(value, (list, tuple, set)):
            return not value or any(TOSLoader._wildcard(item) for item in value)
        return False

    @staticmethod
    def _values_equal(expected: Any, actual: Any) -> bool:
        if isinstance(expected, bool) or isinstance(actual, bool):
            return bool(expected) == bool(actual)

        expected_number = TOSLoader._coerce_float(expected)
        actual_number = TOSLoader._coerce_float(actual)
        if expected_number is not None and actual_number is not None:
            return expected_number == actual_number

        return str(expected).strip().upper() == str(actual).strip().upper()

    @staticmethod
    def _coerce_float(raw_value: Any) -> Optional[float]:
        if raw_value is None or isinstance(raw_value, bool):
            return None
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_priority(raw_value: Any, default: int) -> int:
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return default

    def _space_group_specificity(self, group: Dict[str, Any]) -> int:
        score = 0
        for key in ("category", "carrierKind", "pod", "vesselId"):
            value = group.get(key)
            if key == "category" and value == -1:
                continue
            if not self._wildcard(value):
                score += 1

        filter_all = group.get("filterAll") or {}
        if isinstance(filter_all, dict):
            for key, value in filter_all.items():
                if key in ("weightMin", "weightMax"):
                    if value is not None:
                        score += 1
                elif not self._wildcard(value):
                    score += 1
        return score

    @staticmethod
    def _normalize_range_list(raw_range_list: Any) -> List[Dict[str, Any]]:
        if not raw_range_list:
            return []
        if not isinstance(raw_range_list, list):
            return []
        return [
            deepcopy(item)
            for item in raw_range_list
            if isinstance(item, dict) and item.get("active", True)
        ]

    def _all_active_space_ranges(
        self,
        group_map: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        ranges: List[Dict[str, Any]] = []
        for group in group_map.values():
            if not isinstance(group, dict):
                continue
            ranges.extend(self._normalize_range_list(group.get("rangeList")))
        return ranges

    @staticmethod
    def _container_demand_by_size(containers: List[Container]) -> Dict[int, int]:
        demand: Dict[int, int] = defaultdict(int)
        for container in containers:
            if container.size == ContainerSize.SIZE_40:
                demand[2] += 1
            elif container.size == ContainerSize.SIZE_45:
                demand[3] += 1
            else:
                demand[1] += 1
        return dict(demand)

    def _range_capacity_for_size(
        self,
        yard: Any,
        ranges: List[Dict[str, Any]],
        size_int: int,
    ) -> int:
        if not ranges:
            return 0

        used: Set[Tuple[Any, ...]] = set()
        total = 0
        for record in self._available_column_records(
            yard=yard,
            size_int=size_int,
            include_ranges=ranges,
            exclude_ranges=None,
        ):
            for tier in record["tiers"]:
                atom = (*record["key"], tier)
                if atom in used:
                    continue
                used.add(atom)
                total += 1
        return total

    def _new_ranges_for_deficit(
        self,
        *,
        yard: Any,
        size_int: int,
        deficit: int,
        excluded_ranges: List[Dict[str, Any]],
        template_ranges: List[Dict[str, Any]],
        next_range_seq: int,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        selected: List[Dict[str, Any]] = []
        added_capacity = 0

        for record in self._available_column_records(
            yard=yard,
            size_int=size_int,
            include_ranges=None,
            exclude_ranges=excluded_ranges,
        ):
            selected.append(record)
            added_capacity += len(record["tiers"])
            if added_capacity >= deficit:
                break

        if not selected:
            return [], 0, next_range_seq

        ranges = self._records_to_ranges(
            selected,
            template_ranges=template_ranges,
            next_range_seq=next_range_seq,
        )
        return ranges, added_capacity, next_range_seq + len(ranges)

    def _available_column_records(
        self,
        *,
        yard: Any,
        size_int: int,
        include_ranges: Optional[List[Dict[str, Any]]],
        exclude_ranges: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        large = size_int in (2, 3)

        for (block_id, bay_idx, stack_idx), stack_info in yard.stacks.items():
            if large:
                record = self._large_column_record(
                    yard,
                    block_id,
                    bay_idx,
                    stack_idx,
                    stack_info,
                    size_int,
                )
            else:
                record = self._single_column_record(
                    block_id,
                    bay_idx,
                    stack_idx,
                    stack_info,
                    size_int,
                )

            if record is None:
                continue
            if include_ranges is not None:
                record["tiers"] = [
                    tier
                    for tier in record["tiers"]
                    if self._record_tier_in_ranges(record, tier, include_ranges)
                ]
                if not record["tiers"]:
                    continue
            if exclude_ranges and self._record_overlaps_ranges(record, exclude_ranges):
                continue
            records.append(record)

        records.sort(
            key=lambda item: (
                item["blockId"],
                item["startBayIndex"],
                item["endBayIndex"],
                item["stackIndex"],
            )
        )
        return records

    def _single_column_record(
        self,
        block_id: str,
        bay_idx: int,
        stack_idx: int,
        stack_info: Dict[str, Any],
        size_int: int,
    ) -> Optional[Dict[str, Any]]:
        tiers = self._remaining_20ft_tiers(stack_info)
        if not tiers:
            return None
        return {
            "key": (size_int, block_id, int(bay_idx), int(stack_idx)),
            "blockId": block_id,
            "startBayIndex": int(bay_idx),
            "endBayIndex": int(bay_idx),
            "stackIndex": int(stack_idx),
            "tiers": tiers,
        }

    def _large_column_record(
        self,
        yard: Any,
        block_id: str,
        bay_idx: int,
        stack_idx: int,
        stack_info: Dict[str, Any],
        size_int: int,
    ) -> Optional[Dict[str, Any]]:
        tiers = self._remaining_large_tiers(yard, stack_info)
        if not tiers:
            return None

        first_tier = min(tiers)
        slot_40ft = (stack_info.get("tiers") or {}).get(first_tier, {}).get("slot_40ft")
        display_bays = self._display_bays_for_40ft_slot(
            yard,
            slot_40ft,
            fallback_bay=int(bay_idx),
        )
        return {
            "key": (size_int, block_id, int(bay_idx), int(stack_idx)),
            "blockId": block_id,
            "startBayIndex": min(display_bays),
            "endBayIndex": max(display_bays),
            "stackIndex": int(stack_idx),
            "tiers": tiers,
        }

    @staticmethod
    def _remaining_20ft_tiers(stack_info: Dict[str, Any]) -> List[int]:
        return YardSpaceAdapter._remaining_20ft_tiers(stack_info)

    @staticmethod
    def _remaining_large_tiers(
        yard: Any,
        stack_info: Dict[str, Any],
    ) -> List[int]:
        return YardSpaceAdapter._remaining_large_tiers(yard, stack_info)

    @staticmethod
    def _large_tier_supported(yard: Any, slot_40ft: str, tier: int) -> bool:
        if tier <= 1:
            return True
        slot_info = getattr(yard, "slots_40ft", {}).get(slot_40ft, {})
        for related_name in slot_info.get("related_20ft", []):
            related = getattr(yard, "slots_20ft", {}).get(related_name)
            if not related:
                continue
            related_stack = yard.stacks.get(
                (
                    related["blockId"],
                    related["bayIdx"],
                    related["stackIdx"],
                )
            )
            if related_stack and int(related_stack.get("top_occupied_tier") or 0) < tier - 1:
                return False
        return True

    @staticmethod
    def _display_bays_for_40ft_slot(
        yard: Any,
        slot_40ft: Optional[str],
        fallback_bay: int,
    ) -> Tuple[int, int]:
        if not slot_40ft:
            return fallback_bay, fallback_bay
        slot_info = getattr(yard, "slots_40ft", {}).get(slot_40ft, {})
        bay_numbers: List[int] = []
        for related_name in slot_info.get("related_20ft", []):
            related = getattr(yard, "slots_20ft", {}).get(related_name)
            if related:
                bay_numbers.append(int(related["bayIdx"]))
        if not bay_numbers:
            return fallback_bay, fallback_bay
        return min(bay_numbers), max(bay_numbers)

    def _record_tier_in_ranges(
        self,
        record: Dict[str, Any],
        tier: int,
        ranges: List[Dict[str, Any]],
    ) -> bool:
        return any(self._range_contains_record_tier(range_item, record, tier) for range_item in ranges)

    def _record_overlaps_ranges(
        self,
        record: Dict[str, Any],
        ranges: List[Dict[str, Any]],
    ) -> bool:
        for tier in record["tiers"]:
            if self._record_tier_in_ranges(record, tier, ranges):
                return True
        return False

    @staticmethod
    def _range_contains_record_tier(
        range_item: Dict[str, Any],
        record: Dict[str, Any],
        tier: int,
    ) -> bool:
        if range_item.get("blockId") != record["blockId"]:
            return False

        try:
            start_bay = int(range_item.get("startBayIndex"))
            end_bay = int(range_item.get("endBayIndex"))
            start_stack = int(range_item.get("startStackIndex"))
            end_stack = int(range_item.get("endStackIndex"))
            start_tier = int(range_item.get("startTierIndex", 1))
            end_tier = int(range_item.get("endTierIndex", MAX_TIERS_PER_COLUMN))
        except (TypeError, ValueError):
            return False

        bay_start = min(start_bay, end_bay)
        bay_end = max(start_bay, end_bay)
        return (
            bay_start <= record["startBayIndex"]
            and record["endBayIndex"] <= bay_end
            and min(start_stack, end_stack) <= record["stackIndex"] <= max(start_stack, end_stack)
            and min(start_tier, end_tier) <= int(tier) <= max(start_tier, end_tier)
        )

    def _records_to_ranges(
        self,
        records: List[Dict[str, Any]],
        *,
        template_ranges: List[Dict[str, Any]],
        next_range_seq: int,
    ) -> List[Dict[str, Any]]:
        if not records:
            return []

        ranges: List[Dict[str, Any]] = []
        sorted_records = sorted(
            records,
            key=lambda item: (
                item["blockId"],
                item["startBayIndex"],
                item["endBayIndex"],
                item["stackIndex"],
            ),
        )

        current = sorted_records[0].copy()
        current_start_stack = current["stackIndex"]
        current_end_stack = current["stackIndex"]

        for record in sorted_records[1:]:
            same_bay = (
                record["blockId"] == current["blockId"]
                and record["startBayIndex"] == current["startBayIndex"]
                and record["endBayIndex"] == current["endBayIndex"]
            )
            if same_bay and record["stackIndex"] == current_end_stack + 1:
                current_end_stack = record["stackIndex"]
                continue

            ranges.append(
                self._build_new_range_item(
                    current,
                    current_start_stack,
                    current_end_stack,
                    template_ranges,
                    next_range_seq + len(ranges),
                )
            )
            current = record.copy()
            current_start_stack = current["stackIndex"]
            current_end_stack = current["stackIndex"]

        ranges.append(
            self._build_new_range_item(
                current,
                current_start_stack,
                current_end_stack,
                template_ranges,
                next_range_seq + len(ranges),
            )
        )
        return ranges

    @staticmethod
    def _build_new_range_item(
        record: Dict[str, Any],
        start_stack: int,
        end_stack: int,
        template_ranges: List[Dict[str, Any]],
        range_seq: int,
    ) -> Dict[str, Any]:
        template = template_ranges[0] if template_ranges else {}
        return {
            "blockId": record["blockId"],
            "startBayIndex": int(record["startBayIndex"]),
            "endBayIndex": int(record["endBayIndex"]),
            "startStackIndex": int(start_stack),
            "endStackIndex": int(end_stack),
            "startTierIndex": 1,
            "endTierIndex": MAX_TIERS_PER_COLUMN,
            "slotSize": template.get("slotSize", 1),
            "fillSeq": template.get("fillSeq", 0),
            "active": template.get("active", True),
            "mixWeightType": template.get("mixWeightType", -1),
            "prioritySeq": template.get("prioritySeq", 0),
            "rangeSeq": range_seq,
            "leaveKeySlot": template.get("leaveKeySlot", False),
        }

    @staticmethod
    def _next_range_seq(ranges: List[Dict[str, Any]]) -> int:
        max_seq = 0
        for range_item in ranges:
            try:
                max_seq = max(max_seq, int(range_item.get("rangeSeq", 0)))
            except (TypeError, ValueError):
                continue
        return max_seq + 1

    def load_external_allocation_groups(
        self,
        line_keys: Optional[int],
        vessel_key: Optional[int],
        vessels: Dict[str, Vessel],
    ) -> List[AllocationGroup]:
        normalized_line_key = self._coerce_line_key(line_keys)
        normalized_vessel_key = self._coerce_line_key(vessel_key)
        target_description = (
            f"visitDbkey={normalized_vessel_key}"
            if normalized_vessel_key is not None
            else f"lineKey={normalized_line_key}"
        )
        raise NotImplementedError(
            "type=2 需要从外部分配组接口读取数据；接口未提供，"
            f"已预留 load_external_allocation_groups({target_description})"
        )

    def build_planning_horizon(
        self,
        vessels: Dict[str, Vessel],
        plan_start_time: Optional[datetime] = None,
        plan_end_time: Optional[datetime] = None,
    ) -> Tuple[datetime, datetime]:
        if plan_start_time is not None and plan_end_time is not None:
            logger.info(
                f"TOSLoader: 规划时间窗口 {plan_start_time} → {plan_end_time}"
            )
            return plan_start_time, plan_end_time

        all_vessels = list(vessels.values())
        if not all_vessels:
            now = datetime.now()
            return now, now + timedelta(hours=48)

        horizon_start = min(vessel.eta for vessel in all_vessels)
        horizon_end = max(vessel.etd for vessel in all_vessels)
        if horizon_end <= horizon_start:
            horizon_end = horizon_start + timedelta(hours=48)

        logger.info(f"TOSLoader: 规划时间窗口 {horizon_start} → {horizon_end}")
        return horizon_start, horizon_end

    @staticmethod
    def _visit_overlaps_plan_window(
        eta: Optional[datetime],
        etd: Optional[datetime],
        plan_start: datetime,
        plan_end: datetime,
    ) -> bool:
        """航次 [eta, etd] 与手动规划范围有交集则纳入。"""
        if plan_end <= plan_start:
            return False
        visit_start = eta or plan_start
        if etd is not None:
            visit_end = etd
        elif plan_start <= visit_start < plan_end:
            visit_end = plan_end
        else:
            return False
        return visit_start < plan_end and visit_end > plan_start

    @staticmethod
    def _parse_dt(dt_str: Optional[str]) -> Optional[datetime]:
        if not dt_str:
            return None

        cleaned = dt_str.split("+")[0].split("Z")[0].strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
        logger.warning(f"TOSLoader: 无法解析时间字符串: {dt_str!r}")
        return None

    @classmethod
    def _coerce_line_key_set(cls, raw_values: Any) -> Set[int]:
        if raw_values is None:
            return set()

        values = (
            raw_values
            if isinstance(raw_values, (list, tuple, set))
            else [raw_values]
        )
        return {
            line_key
            for line_key in (cls._coerce_line_key(value) for value in values)
            if line_key is not None
        }

    @staticmethod
    def _coerce_line_key(raw_value: Any) -> Optional[int]:
        if raw_value is None or isinstance(raw_value, bool):
            return None
        if isinstance(raw_value, int):
            return raw_value
        if isinstance(raw_value, float):
            return int(raw_value) if raw_value.is_integer() else None

        text = str(raw_value).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @staticmethod
    def _coerce_bool(raw_value: Any) -> bool:
        if isinstance(raw_value, bool):
            return raw_value
        if raw_value is None:
            return False
        if isinstance(raw_value, (int, float)):
            return raw_value != 0
        return str(raw_value).strip().lower() in {"1", "true", "yes", "y"}

    @classmethod
    def _parse_size(cls, size_int: int) -> ContainerSize:
        return cls._SIZE_MAP.get(size_int, ContainerSize.SIZE_20)

    @classmethod
    def _parse_container_type(
        cls,
        iso_code: str,
        power_required: int,
    ) -> ContainerType:
        if power_required:
            return ContainerType.REEFER
        code = (iso_code or "").upper()
        for suffix, container_type in cls._ISO_TYPE_MAP.items():
            if suffix in code:
                return container_type
        return ContainerType.DRY

    @classmethod
    def _parse_weight_class(
        cls,
        weight: Optional[float],
        freight_kind: Optional[int],
    ) -> WeightClass:
        if freight_kind == 3:
            return WeightClass.EMPTY
        if weight is None or weight <= 0:
            return WeightClass.EMPTY
        if weight > cls.HEAVY_WEIGHT_THRESHOLD:
            return WeightClass.HEAVY
        return WeightClass.LIGHT


__all__ = ["YardSpaceAdapter", "TOSLoader"]
