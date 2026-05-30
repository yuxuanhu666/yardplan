from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

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
                    if YardSpaceAdapter._is_empty_20ft_column(stack_info):
                        stage2_single_slots.append(
                            {
                                "bay_number": bay_number,
                                "stack_index": int(stack_key[2]),
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
                        if not YardSpaceAdapter._is_empty_40ft_column(stack_info):
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
                                "is_edge_pair": is_edge,
                            }
                        )

            area = YardArea(
                area_id=block_id,
                business_type=business_type,
                bays=bays,
                large_bay_pairs=large_bay_pairs,
                max_stack_height=MAX_TIERS_PER_COLUMN,
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
        token: Optional[str] = None,
    ):
        self.token = token
        self.vessel_visit_path = vessel_visit_path or os.path.join(
            _DATA_DIR,
            "217getVesselVisit（船舶访问计划）.json",
        )
        self.bound_list_path = bound_list_path or os.path.join(
            _DATA_DIR,
            "217getBoundList（装船箱和卸船箱列表）.json",
        )

    def load_vessels(
        self,
        line_keys: List[int],
        plan_start_time: Optional[datetime] = None,
        plan_end_time: Optional[datetime] = None,
    ) -> Dict[str, Vessel]:
        with open(self.vessel_visit_path, "r", encoding="utf-8") as file:
            raw: List[dict] = json.load(file)

        target_set = {
            line_key
            for line_key in (self._coerce_line_key(value) for value in line_keys)
            if line_key is not None
        }
        vessels: Dict[str, Vessel] = {}
        matched_line_keys: Set[int] = set()
        skipped_by_plan_window = 0

        for item in raw:
            vessel_visit_id = item.get("vesselVisitId", "")
            line_key = self._coerce_line_key(item.get("lineKey"))
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

            vessels[vessel_visit_id] = Vessel(
                vessel_id=vessel_id_str,
                vessel_name=vessel_name,
                voyage_id=vessel_visit_id,
                eta=eta or datetime.now(),
                etd=etd or (datetime.now() + timedelta(hours=48)),
                berth_id="",
            )

        missing = target_set - matched_line_keys
        if missing:
            logger.warning(f"TOSLoader: VesselVisit 中未找到航线号: {sorted(missing)}")
        if plan_start_time is not None and plan_end_time is not None:
            logger.info(
                f"TOSLoader: 规划范围 {plan_start_time} → {plan_end_time}，"
                f"加载船舶 {len(vessels)} 艘"
                f" (lineKeys: {sorted(target_set)}，范围外跳过 {skipped_by_plan_window} 艘)"
            )
        else:
            logger.info(
                f"TOSLoader: 加载船舶 {len(vessels)} 艘"
                f" (lineKeys: {sorted(target_set)})"
            )
        return vessels

    def load_discharge_containers(
        self,
        line_keys: List[int],
        vessels: Dict[str, Vessel],
    ) -> List[Container]:
        return self._load_bound_containers(
            line_keys=line_keys,
            vessels=vessels,
            section_name="Inbound",
            business_type=BusinessType.IMPORT,
            expected_bound_type=1,
            expected_visit_type=1,
            description="卸船箱",
        )

    def load_loading_containers(
        self,
        line_keys: List[int],
        vessels: Dict[str, Vessel],
    ) -> List[Container]:
        # 出口箱：217getBoundList 的 Outbound；当前数据里 boundType/visitType 与 Inbound 一致，均为 (1,1)。
        return self._load_bound_containers(
            line_keys=line_keys,
            vessels=vessels,
            section_name="Outbound",
            business_type=BusinessType.EXPORT,
            expected_bound_type=1,
            expected_visit_type=1,
            description="装船箱",
        )

    def _load_bound_containers(
        self,
        line_keys: List[int],
        vessels: Dict[str, Vessel],
        section_name: str,
        business_type: BusinessType,
        expected_bound_type: int,
        expected_visit_type: int,
        description: str,
    ) -> List[Container]:
        with open(self.bound_list_path, "r", encoding="utf-8") as file:
            raw: dict = json.load(file)

        entries: List[dict] = raw.get(section_name, [])
        target_set = {
            line_key
            for line_key in (self._coerce_line_key(value) for value in line_keys)
            if line_key is not None
        }
        containers: List[Container] = []

        for item in entries:
            container_raw = item.get("container")
            if not container_raw:
                continue

            service_line_key = self._coerce_line_key(
                container_raw.get("serviceLineKey")
            )
            if service_line_key not in target_set:
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

        logger.info(
            f"TOSLoader: 加载{description} {len(containers)} 个"
            f" (lineKeys: {sorted(target_set)})"
        )
        return containers

    def load_external_allocation_groups(
        self,
        line_keys: List[int],
        vessels: Dict[str, Vessel],
    ) -> List[AllocationGroup]:
        normalized_line_keys = [
            line_key
            for line_key in (self._coerce_line_key(value) for value in line_keys)
            if line_key is not None
        ]
        raise NotImplementedError(
            "type=2 需要从外部分配组接口读取数据；接口未提供，"
            f"已预留 load_external_allocation_groups(line_keys={normalized_line_keys})"
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
