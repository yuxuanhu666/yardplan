from __future__ import annotations

import os
from datetime import datetime
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
    return plan_start_time, plan_end_time


def _range_plan_data(result: PlanningResult) -> List[Dict[str, Any]]:
    return result.metrics.get("range_plan", {}).get("data", [])


def run_plan(
    token: Optional[str] = None,
    *,
    line_keys: List[int],
    type: int = 1,
    apply_to_yard: bool = False,
    save_visualization: bool = False,
    visualization_path: Optional[str] = None,
    visualization_block_ids: Optional[Sequence[str]] = None,
    visualization_use_real_coordinates: bool = True,
    plan_start_time: Optional[datetime] = None,
    plan_end_time: Optional[datetime] = None,
    return_list: bool = False,
) -> Union[PlanningResult, List[Dict[str, Any]]]:
    """
    规划入口：传入一个或多个航线号，完成进出口箱堆场分配规划。

    token: 第一个参数，TOS/平台访问令牌（str），默认 None；不传时由调用方（如 FastAPI）决定是否回退到服务注册 token。
    line_keys: 航线号列表（keyword-only）。
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
    normalized_line_keys: List[int] = []
    for raw_key in line_keys:
        line_key = loader._coerce_line_key(raw_key)
        if line_key is None:
            raise ValueError(f"无效的 lineKey: {raw_key!r}")
        normalized_line_keys.append(line_key)

    print("=" * 70)
    print(f"  堆场规划  type={type}  lineKeys: {normalized_line_keys}")
    if plan_start_time is not None and plan_end_time is not None:
        print(f"  规划范围: {plan_start_time} → {plan_end_time}")
    print("=" * 70)

    vessels = loader.load_vessels(
        normalized_line_keys,
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
            normalized_line_keys,
            vessels,
        )
        export_containers = loader.load_loading_containers(
            normalized_line_keys,
            vessels,
        )
        containers = import_containers + export_containers
        if not containers:
            logger.warning("未加载到任何进出口箱，规划终止")
            return PlanningResult(
                run_id="PLAN-EMPTY",
                timestamp=datetime.now(),
                mode=PlannerMode.FULL_PLAN,
            )
    else:
        external_groups = loader.load_external_allocation_groups(
            normalized_line_keys,
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
            mode=PlannerMode.FULL_PLAN,
            apply_to_yard=apply_to_yard,
            horizon_start=horizon_start,
            horizon_end=horizon_end,
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
        )

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
