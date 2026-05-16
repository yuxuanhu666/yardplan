from __future__ import annotations

import hashlib
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from yardplan_core.models import (
    AllocationGroup,
    BusinessType,
    Vessel,
    YardArea,
    logger,
)

if TYPE_CHECKING:
    from yardplan_core.allocation import TimeStep


@dataclass(frozen=True)
class AreaStepWorkload:
    """
    某次规划时刻下，单个箱区在单个时间步上的预计场桥作业量（箱次 / moves）。

    - `inbound_moves`: 进入箱区的作业次数，例如卸船落场、收箱落场。
    - `outbound_moves`: 离开箱区的作业次数，例如提箱装船、闸口提箱。
    - `max_moves`: 该箱区该时间步的理论作业能力上限。
    """

    area_id: str
    step_id: int
    step_start: datetime
    step_end: datetime
    inbound_moves: float
    outbound_moves: float
    max_moves: float
    source: str

    @property
    def total_moves(self) -> float:
        return self.inbound_moves + self.outbound_moves

    @property
    def utilization_ratio(self) -> float:
        """`total_moves / max_moves`，裁剪到 `[0, 1]`。"""
        if self.max_moves <= 0:
            return 0.0
        return max(0.0, min(1.0, self.total_moves / self.max_moves))


@dataclass
class AreaWorkloadSnapshot:
    """
    某次规划时刻的箱区工作量快照。

    `target_step_ids` 记录每条船 `ETA` 落入的目标时间步。Stage1 默认只在这些 ETA
    目标步上做工作量均衡，而不是用整个规划期的列利用率 spread 近似替代作业量。

    业务定义上，场桥工作量必须用 moves（箱次）衡量：
    - 基线工作量：来自 `(area_id, step_id)` 上已有 `inbound_moves/outbound_moves`
    - 计划增量：来自 Stage1 分配组的 `container_count`，按 ETA~ETD 覆盖步数均摊
    禁止再用 `column_demand`（列）作为 workload 度量。
    """

    time_step_hours: float
    steps: List["TimeStep"]
    by_area_step: Dict[Tuple[str, int], AreaStepWorkload]
    target_step_ids: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def step_for_voyage(self, voyage_id: str) -> Optional[int]:
        return self.target_step_ids.get(voyage_id)


@dataclass
class WorkloadEstimationConfig:
    time_step_hours: float = 4.0
    rtg_count_per_area: int = 2
    moves_per_rtg_per_hour: int = 30
    planning_lead_days: int = 7
    random_seed: int = 17
    inventory_activity_floor: float = 0.05
    inventory_activity_cap: float = 0.85
    eta_activity_boost: float = 1.25
    noise_sigma: float = 0.12
    low_activity_probability: float = 0.25

    @property
    def max_moves_per_step(self) -> float:
        return (
            self.rtg_count_per_area
            * self.moves_per_rtg_per_hour
            * self.time_step_hours
        )


class AreaWorkloadProvider(ABC):
    @abstractmethod
    def build_snapshot(
        self,
        *,
        yard_areas: List[YardArea],
        time_steps: List["TimeStep"],
        vessels: Dict[str, Vessel],
        groups: List[AllocationGroup],
        plan_start_time: datetime,
        plan_end_time: datetime,
        config: WorkloadEstimationConfig,
    ) -> AreaWorkloadSnapshot:
        raise NotImplementedError


class TOSAreaWorkloadProvider(AreaWorkloadProvider):
    """
    未来从 TOS 读取 `(area_id, step_id) -> moves`。

    典型实现方向：
    1. 将 TOS 的卸船、装船、收箱、提箱事件统一映射到时间步。
    2. 输出每个箱区在每个时间步上的 `inbound_moves / outbound_moves`。
    3. 结合箱区 RTG 配置推导 `max_moves`。
    """

    def build_snapshot(
        self,
        *,
        yard_areas: List[YardArea],
        time_steps: List["TimeStep"],
        vessels: Dict[str, Vessel],
        groups: List[AllocationGroup],
        plan_start_time: datetime,
        plan_end_time: datetime,
        config: WorkloadEstimationConfig,
    ) -> AreaWorkloadSnapshot:
        raise NotImplementedError("TOSAreaWorkloadProvider 尚未实现")


class FileAreaWorkloadProvider(AreaWorkloadProvider):
    """
    未来从文件读取箱区时间步工作量。

    JSON 示例：
    {
      "timeStepHours": 4.0,
      "items": [
        {
          "areaId": "A01",
          "stepId": 12,
          "inboundMoves": 38,
          "outboundMoves": 91,
          "maxMoves": 240
        }
      ]
    }
    """

    def build_snapshot(
        self,
        *,
        yard_areas: List[YardArea],
        time_steps: List["TimeStep"],
        vessels: Dict[str, Vessel],
        groups: List[AllocationGroup],
        plan_start_time: datetime,
        plan_end_time: datetime,
        config: WorkloadEstimationConfig,
    ) -> AreaWorkloadSnapshot:
        raise NotImplementedError("FileAreaWorkloadProvider 尚未实现")


class SimulatedAreaWorkloadProvider(AreaWorkloadProvider):
    """
    基于当前库存和可复现随机数模拟 `(area, step)` 工作量。

    业务含义上，库存与作业强度解耦：箱区可以库存很高，但在某个 ETA 时间步里实际
    的进出场桥作业仍然偏低。因此模拟既使用 `occupied_columns / total_columns` 作为
    活跃度基线，也保留“高库存但低作业”的低活跃抽样。
    """

    def build_snapshot(
        self,
        *,
        yard_areas: List[YardArea],
        time_steps: List["TimeStep"],
        vessels: Dict[str, Vessel],
        groups: List[AllocationGroup],
        plan_start_time: datetime,
        plan_end_time: datetime,
        config: WorkloadEstimationConfig,
    ) -> AreaWorkloadSnapshot:
        del groups

        target_step_ids = self._build_target_step_ids(time_steps, vessels)
        target_step_counts: Dict[int, int] = {}
        for step_id in target_step_ids.values():
            target_step_counts[step_id] = target_step_counts.get(step_id, 0) + 1

        by_area_step: Dict[Tuple[str, int], AreaStepWorkload] = {}
        for area in yard_areas:
            stock_ratio = area.occupied_columns / max(1, area.total_columns)
            for step in time_steps:
                rng = random.Random(
                    self._derive_seed(config.random_seed, area.area_id, str(step.step_id))
                )
                base = config.inventory_activity_floor + stock_ratio * (
                    config.inventory_activity_cap - config.inventory_activity_floor
                )
                if rng.random() < config.low_activity_probability:
                    base *= rng.uniform(0.05, 0.35)

                eta_count = target_step_counts.get(step.step_id, 0)
                if eta_count > 0:
                    base *= min(config.eta_activity_boost ** eta_count, 2.0)

                gross_factor = max(0.0, base) * (0.5 + 0.5 * rng.random())
                noise_factor = (
                    math.exp(rng.gauss(0.0, config.noise_sigma))
                    if config.noise_sigma > 0
                    else 1.0
                )
                total_moves = config.max_moves_per_step * gross_factor * noise_factor

                inbound_share, outbound_share = self._directional_shares(
                    area.business_type,
                    rng,
                )
                inbound_moves = max(0.0, total_moves * inbound_share)
                outbound_moves = max(0.0, total_moves * outbound_share)

                sum_moves = inbound_moves + outbound_moves
                if sum_moves > config.max_moves_per_step and sum_moves > 0:
                    scale = config.max_moves_per_step / sum_moves
                    inbound_moves *= scale
                    outbound_moves *= scale

                by_area_step[(area.area_id, step.step_id)] = AreaStepWorkload(
                    area_id=area.area_id,
                    step_id=step.step_id,
                    step_start=step.start_time,
                    step_end=step.end_time,
                    inbound_moves=inbound_moves,
                    outbound_moves=outbound_moves,
                    max_moves=config.max_moves_per_step,
                    source="simulated",
                )

        snapshot = AreaWorkloadSnapshot(
            time_step_hours=config.time_step_hours,
            steps=time_steps,
            by_area_step=by_area_step,
            target_step_ids=target_step_ids,
            metadata={
                "provider": "simulated",
                "planning_reference_time": plan_start_time,
                "plan_start_time": plan_start_time,
                "plan_end_time": plan_end_time,
                "planning_lead_days": config.planning_lead_days,
                "random_seed": config.random_seed,
                "voyage_eta": {
                    voyage_id: vessel.eta for voyage_id, vessel in vessels.items()
                },
            },
        )
        self._log_snapshot(snapshot, yard_areas)
        return snapshot

    @staticmethod
    def _directional_shares(
        business_type: BusinessType,
        rng: random.Random,
    ) -> Tuple[float, float]:
        primary_share = min(0.85, max(0.55, 0.7 + rng.uniform(-0.08, 0.08)))
        secondary_share = 1.0 - primary_share
        if business_type == BusinessType.EXPORT:
            return secondary_share, primary_share
        return primary_share, secondary_share

    @staticmethod
    def _derive_seed(parent_seed: int, area_id: str, step_id: str) -> int:
        payload = f"{parent_seed}:{area_id}:{step_id}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    @staticmethod
    def _build_target_step_ids(
        time_steps: List["TimeStep"],
        vessels: Dict[str, Vessel],
    ) -> Dict[str, int]:
        target_step_ids: Dict[str, int] = {}
        for voyage_id, vessel in vessels.items():
            step = _find_step_containing(time_steps, vessel.eta)
            if step is not None:
                target_step_ids[voyage_id] = step.step_id
        return target_step_ids

    def _log_snapshot(
        self,
        snapshot: AreaWorkloadSnapshot,
        yard_areas: List[YardArea],
    ) -> None:
        print_eta_step_workloads(
            snapshot,
            yard_areas,
            title="【工作量模拟】ETA 目标时间步 · 各箱区场桥工作量（箱次；规划前基线）",
        )

        sample_area_ids = [area.area_id for area in yard_areas[:3]]
        for area_id in sample_area_ids:
            series = [
                snapshot.by_area_step[(area_id, step.step_id)].total_moves
                for step in snapshot.steps
                if (area_id, step.step_id) in snapshot.by_area_step
            ]
            logger.debug("Workload sample area=%s totals=%s", area_id, series)


def _step_by_id(steps: List["TimeStep"], step_id: int) -> Optional["TimeStep"]:
    for step in steps:
        if step.step_id == step_id:
            return step
    return None


def print_eta_step_workloads(
    snapshot: AreaWorkloadSnapshot,
    yard_areas: List[YardArea],
    *,
    title: str = "ETA 目标时间步 · 各箱区场桥工作量（箱次）",
    resolve_workload: Optional[
        Callable[[str, int], Tuple[float, float, float, float]]
    ] = None,
) -> None:
    """
    在控制台打印每条船 ETA 所在时间步下，各箱区的进/出/总作业量与能力利用率。

    这里的“工作量”单位始终是箱次（moves），不是列。若传入 `resolve_workload`，
    它通常会返回“基线 + Stage1 计划增量”后的值，其中计划箱数可按 ETA~ETD
    覆盖的时间步数均摊到 ETA 目标步显示。

    `resolve_workload(area_id, step_id)` 可返回 `(inbound, outbound, total, max_moves)`；
    未提供时直接使用快照中的模拟基线值。
    """
    if not snapshot.target_step_ids:
        print(f"\n{title}\n  （无船舶 ETA 落入规划时间步，跳过）\n")
        return

    voyage_eta = snapshot.metadata.get("voyage_eta", {})
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)

    for voyage_id, step_id in sorted(snapshot.target_step_ids.items()):
        eta = voyage_eta.get(voyage_id)
        step = _step_by_id(snapshot.steps, step_id)
        step_range = (
            f"{step.start_time} ~ {step.end_time}"
            if step is not None
            else f"step_id={step_id}"
        )
        print(
            f"\n航次 {voyage_id}  |  ETA={eta}  |  目标步 step_id={step_id}  |  {step_range}"
        )
        print(
            f"{'箱区':<10} {'业态':<8} {'进箱':>8} {'出箱':>8} {'合计':>8} {'上限':>8} {'利用率':>8}"
        )
        print("-" * 72)

        rows: List[Tuple[str, str, float, float, float, float, float]] = []
        for area in sorted(yard_areas, key=lambda item: item.area_id):
            if resolve_workload is not None:
                inbound, outbound, total, max_moves = resolve_workload(
                    area.area_id,
                    step_id,
                )
            else:
                workload = snapshot.by_area_step.get((area.area_id, step_id))
                if workload is None:
                    continue
                inbound = workload.inbound_moves
                outbound = workload.outbound_moves
                total = workload.total_moves
                max_moves = workload.max_moves
            ratio = total / max_moves if max_moves > 0 else 0.0
            business = (
                "进口"
                if area.business_type == BusinessType.IMPORT
                else "出口"
            )
            rows.append(
                (area.area_id, business, inbound, outbound, total, max_moves, ratio)
            )

        for area_id, business, inbound, outbound, total, max_moves, ratio in rows:
            print(
                f"{area_id:<10} {business:<8} {inbound:8.1f} {outbound:8.1f} "
                f"{total:8.1f} {max_moves:8.1f} {ratio:8.3f}"
            )

        if rows:
            totals = [row[4] for row in rows]
            business_spreads: List[str] = []
            for business_name in ("进口", "出口"):
                business_totals = [row[4] for row in rows if row[1] == business_name]
                if business_totals:
                    business_spreads.append(
                        f"{business_name}spread={max(business_totals) - min(business_totals):.1f}"
                    )
            print("-" * 72)
            print(
                f"{'合计':<10} {'':<8} {'':>8} {'':>8} "
                f"{sum(totals):8.1f} {'':>8} {'  '.join(business_spreads)}"
            )

    print("=" * 88 + "\n")


def _find_step_containing(
    steps: List["TimeStep"],
    point_in_time: datetime,
) -> Optional["TimeStep"]:
    for index, step in enumerate(steps):
        is_last = index == len(steps) - 1
        if step.start_time <= point_in_time < step.end_time:
            return step
        if is_last and point_in_time == step.end_time:
            return step
    return None


__all__ = [
    "AreaStepWorkload",
    "AreaWorkloadSnapshot",
    "WorkloadEstimationConfig",
    "AreaWorkloadProvider",
    "SimulatedAreaWorkloadProvider",
    "TOSAreaWorkloadProvider",
    "FileAreaWorkloadProvider",
    "print_eta_step_workloads",
]
