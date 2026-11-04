"""Run the deterministic paper example and export machine-readable results."""
import argparse
import csv
import json
import math
from pathlib import Path

from algorithm_3 import analyze_algorithm_3
from dag_baseline import (DAG, assign_priorities, divide_subgraphs,
                          graham_bound, local_paths, simulate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).parent / "examples/paper_12_nodes.json")
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("outputs/paper_demo"))
    parser.add_argument("--thermal", action="store_true", help="Export synthetic thermal data")
    parser.add_argument("--seconds-per-unit", type=float,
                        help="Explicit conversion from schedule units to seconds")
    parser.add_argument("--thermal-dt", type=float, default=1.0)
    parser.add_argument("--thermal-seed", type=int, default=42)
    parser.add_argument("--coupling", type=float, default=0.0)
    args = parser.parse_args()
    if args.thermal and args.seconds_per_unit is None:
        parser.error("--thermal requires --seconds-per-unit; the paper example uses abstract units")
    try:
        data = json.loads(args.input.read_text())
        dag = DAG({int(v): c for v, c in data["wcet"].items()}, data["edges"])
        path, length = dag.critical_path()
        joins, blocks, tail = divide_subgraphs(dag)
        priorities = assign_priorities(dag, blocks, tail)
        events = simulate(dag, priorities, args.cores)
        paths = [local_paths(dag, block) for block in blocks]
        observed = max(event["finish"] for event in events)
        analysis = analyze_algorithm_3(dag, blocks, tail, paths, priorities, args.cores,
                                       observed_makespan=observed)
        period, deadline = data.get("period"), data.get("deadline")
        if (period is None) != (deadline is None):
            raise ValueError("Supply both period and deadline, or neither")
        if period is not None and (not math.isfinite(period) or period <= 0 or deadline != period):
            raise ValueError("This baseline requires finite positive period and implicit deadline D=T")
        report = {
            "input": str(args.input.resolve()), "time_unit": data.get("time_unit", "unspecified"),
            "cores": args.cores, "workload": dag.workload(),
            "critical_path": path, "critical_path_length": length,
            "graham_bound": graham_bound(dag, args.cores),
            "join_nodes": joins, "blocks": blocks, "tail": tail,
            "local_paths": paths,
            "priorities": priorities,
            "simulated_makespan": observed,
            "analytical_algorithm_3_bound": analysis["bound"],
            "algorithm_3_status": analysis["status"],
            "model_safe_bound": analysis["model_safe_bound"],
            "algorithm_3_details": analysis,
            "period": period, "deadline": deadline,
            "isolated_release_bound_meets_deadline": (analysis["model_safe_bound"] <= deadline
                                                     if deadline is not None else None),
            "generation": data.get("generation"), "seed": data.get("seed"),
            "notes": ["One release, all durations equal WCET, no communication cost.",
                      "Simulated makespan is NOT a worst-case response-time guarantee.",
                      "Algorithm 3 is experimental: known counterexamples prevent using it as a guarantee.",
                      "Use model_safe_bound (Graham) for the deadline comparison.",
                      "Deadline comparison uses an isolated release; overlapping jobs are not simulated.",
                      "Synthetic thermal data is optional; thermal control is not implemented."],
        }
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "schedule.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    if args.thermal:
        from thermal_model import generate_thermal_trace
        try:
            generate_thermal_trace(args.out / "schedule.csv", args.out / "thermal_trace.csv",
                                   cores=args.cores, seconds_per_unit=args.seconds_per_unit,
                                   dt=args.thermal_dt, seed=args.thermal_seed,
                                   coupling=args.coupling, plot_path=args.out / "thermal_trace.png")
        except (ValueError, TypeError, OSError) as error:
            parser.error(str(error))
    print(json.dumps(report, indent=2))
    print(f"\nSaved report.json and schedule.csv in {args.out.resolve()}")


if __name__ == "__main__":
    main()
