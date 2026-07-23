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
    import json
    from datetime import datetime
    from yardplan_score import format_score_report

    default_start = datetime.fromisoformat("2026-03-25T00:00:00")
    default_end = datetime.fromisoformat("2026-04-10T00:00:00")

    # 航次 YH25004（远航888）：ETA 2025-12-21 16:00，ETD 2025-12-25 16:00
    parser = argparse.ArgumentParser(description="Local yard planning test runner")
    parser.add_argument(
        "--request-json",
        type=str,
        help="Path to a JSON file using the same body as the FastAPI endpoint.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--line-key",
        dest="line_keys",
        type=int,
        help="Service line key.",
    )
    target.add_argument(
        "--vessel-key",
        dest="vessel_key",
        type=int,
        help="VesselVisit dbkey.",
    )
    parser.add_argument("--type", dest="plan_type", type=int, default=1)
    parser.add_argument(
        "--start",
        type=str,
        default=default_start.isoformat(),
        help="Plan start time, e.g. 2025-12-21T00:00:00",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=default_end.isoformat(),
        help="Plan end time, e.g. 2025-12-26T00:00:00",
    )
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
    parser.add_argument(
        "--plot-convergence",
        dest="plot_convergence",
        action="store_true",
        default=False,
        help="Save Stage1 convergence and Stage2 greedy progress chart.",
    )
    parser.add_argument(
        "--no-plot-convergence",
        dest="plot_convergence",
        action="store_false",
        help="Skip convergence/progress chart generation.",
    )
    parser.add_argument(
        "--convergence-path",
        type=str,
        default="outputs/convergence.svg",
        help="Convergence chart output path, e.g. outputs/convergence.svg.",
    )
    args = parser.parse_args()

    result = None
    if args.request_json:
        with open(args.request_json, "r", encoding="utf-8") as request_file:
            request_body = json.load(request_file)
        if not isinstance(request_body, dict):
            raise ValueError("The request JSON root must be an object")
        result = run_plan(
            vessel_line_keys=request_body.get("vesselLineKeyList"),
            vessel_visit_keys=request_body.get("vesselVisitKeyList"),
            import_info=request_body.get("importInfo"),
            export_info=request_body.get("exportInfo"),
            is_auto_group=request_body.get("isAutoGroup"),
            is_auto_range=request_body.get("isAutoRange"),
            save_visualization=args.visualize,
            visualization_path=args.visualization_path,
            print_score=True,
        )
    else:
        plan_start_time = (
            datetime.fromisoformat(args.start) if args.start else default_start
        )
        plan_end_time = datetime.fromisoformat(args.end) if args.end else default_end
        if args.line_keys:
            result = run_plan(
                line_keys=args.line_keys,
                type=args.plan_type,
                save_visualization=args.visualize,
                visualization_path=args.visualization_path,
                plan_start_time=plan_start_time,
                plan_end_time=plan_end_time,
                print_score=True,
            )
        else:
            result = run_plan(
                vessel_key=args.vessel_key or 15623707,
                type=args.plan_type,
                save_visualization=args.visualize,
                visualization_path=args.visualization_path,
                plan_start_time=plan_start_time,
                plan_end_time=plan_end_time,
                print_score=True,
            )

    score = getattr(result, "metrics", {}).get("score") if result is not None else None
    if score:
        print("\nFinal score:")
        print(format_score_report(score, indent="  "))

    if args.plot_convergence and result is not None:
        from yardplan_convergence_plot import plot_convergence_report

        output = plot_convergence_report(
            result,
            args.convergence_path,
            title="Yard Planning Algorithm Progress",
        )
        print(f"\nConvergence chart saved to: {output}")
