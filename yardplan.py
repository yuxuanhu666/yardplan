#!/usr/bin/env python3
"""
Backward-compatible facade for the yard planning package.

External callers can keep importing symbols from `yardplan` and calling
`run_plan()` as before, while the implementation now lives in `yardplan_core`.
"""

from yardplan_core import (
    AllocationEngine,
    AllocationGroup,
    AreaAssignment,
    AreaWorkloadProvider,
    AreaWorkloadSnapshot,
    AreaResourceState,
    AreaScoringStrategy,
    Bay,
    BayColumnAllocation,
    BusinessType,
    ColumnDemandConverter,
    ConstraintChecker,
    Container,
    ContainerSize,
    ContainerType,
    DEFAULT_EXPORT_GROUP_KEYS,
    DEFAULT_IMPORT_GROUP_KEYS,
    DefaultAreaScoringStrategy,
    FlowType,
    GroupKeyFunc,
    GroupingConfig,
    GroupingEngine,
    LargeBayPair,
    MAX_SPLITS_PER_GROUP,
    MAX_TIERS_PER_COLUMN,
    MIN_COLUMNS_PER_SPLIT,
    PlannerMode,
    PlanningResult,
    PlacementPreview,
    ResultFormatter,
    RollingWindow,
    RollingWindowPlanner,
    SimulatedAreaWorkloadProvider,
    Stage1LNSConfig,
    Stage1YardAreaAssigner,
    Stage2BayAllocator,
    TOSAreaWorkloadProvider,
    TOSLoader,
    TimeStep,
    Vessel,
    WorkloadEstimationConfig,
    WeightClass,
    YardArea,
    YardFlow,
    YardVisualizationConfig,
    YardVisualizer,
    YardPlanner,
    YardSpaceAdapter,
    _DATA_DIR,
    logger,
    run_plan,
    save_yard_visualization,
    plot_yard,
    build_yard_layout,
)

__all__ = [
    "_DATA_DIR",
    "logger",
    "MAX_TIERS_PER_COLUMN",
    "MAX_SPLITS_PER_GROUP",
    "MIN_COLUMNS_PER_SPLIT",
    "BusinessType",
    "ContainerSize",
    "ContainerType",
    "WeightClass",
    "PlannerMode",
    "FlowType",
    "Container",
    "Bay",
    "LargeBayPair",
    "YardArea",
    "Vessel",
    "YardFlow",
    "YardVisualizationConfig",
    "YardVisualizer",
    "AllocationGroup",
    "AreaAssignment",
    "BayColumnAllocation",
    "PlanningResult",
    "ColumnDemandConverter",
    "GroupKeyFunc",
    "DEFAULT_EXPORT_GROUP_KEYS",
    "DEFAULT_IMPORT_GROUP_KEYS",
    "GroupingConfig",
    "GroupingEngine",
    "TimeStep",
    "RollingWindow",
    "RollingWindowPlanner",
    "ConstraintChecker",
    "PlacementPreview",
    "AreaResourceState",
    "Stage1LNSConfig",
    "AreaScoringStrategy",
    "DefaultAreaScoringStrategy",
    "Stage1YardAreaAssigner",
    "Stage2BayAllocator",
    "AllocationEngine",
    "AreaWorkloadProvider",
    "AreaWorkloadSnapshot",
    "SimulatedAreaWorkloadProvider",
    "TOSAreaWorkloadProvider",
    "WorkloadEstimationConfig",
    "YardSpaceAdapter",
    "TOSLoader",
    "ResultFormatter",
    "YardPlanner",
    "run_plan",
    "save_yard_visualization",
    "plot_yard",
    "build_yard_layout",
]


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    # 航次 YH25004（远航888）：ETA 2025-12-21 16:00，ETD 2025-12-25 16:00
    parser = argparse.ArgumentParser(description="Local yard planning test runner")
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--line-key",
        dest="line_keys",
        action="append",
        type=int,
        help="Line key. Repeat it to pass more than one line key.",
    )
    target.add_argument(
        "--vessel-key",
        dest="vessel_key",
        type=int,
        help="Vessel key. With type=1, only containers for this vessel key are grouped.",
    )
    parser.add_argument("--type", dest="plan_type", type=int, default=1)
    parser.add_argument("--start", type=str, help="Plan start time, e.g. 2025-12-21T00:00:00")
    parser.add_argument("--end", type=str, help="Plan end time, e.g. 2025-12-26T00:00:00")
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        default=True,
        help="Save visualization image (default for local CLI runs).",
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Skip visualization image generation.",
    )
    parser.add_argument(
        "--visualization-path",
        type=str,
        help="Visualization output path, e.g. outputs/line15525576.png.",
    )
    args = parser.parse_args()

    plan_start_time = datetime.fromisoformat(args.start) if args.start else None
    plan_end_time = datetime.fromisoformat(args.end) if args.end else None

    if args.line_keys:
        run_plan(
            line_keys=args.line_keys,
            type=args.plan_type,
            save_visualization=args.visualize,
            visualization_path=args.visualization_path,
            plan_start_time=plan_start_time,
            plan_end_time=plan_end_time,
        )
    else:
        run_plan(
            vessel_key=args.vessel_key or 6861435,
            type=args.plan_type,
            save_visualization=args.visualize,
            visualization_path=args.visualization_path,
            plan_start_time=plan_start_time,
            plan_end_time=plan_end_time,
        )
