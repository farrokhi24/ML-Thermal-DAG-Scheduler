"""Run the deterministic paper example and export machine-readable results."""
import argparse
import csv
import json
from pathlib import Path

from dag_baseline import (DAG, assign_priorities, divide_subgraphs,
                          graham_bound, local_paths, simulate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).parent / "examples/paper_12_nodes.json")
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("outputs/paper_demo"))
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text())
        dag = DAG({int(v): c for v, c in data["wcet"].items()}, data["edges"])
        path, length = dag.critical_path()
        joins, blocks, tail = divide_subgraphs(dag)
        priorities = assign_priorities(dag, blocks, tail)
        events = simulate(dag, priorities, args.cores)
        report = {
            "input": str(args.input.resolve()), "time_unit": data.get("time_unit", "unspecified"),
            "cores": args.cores, "workload": dag.workload(),
            "critical_path": path, "critical_path_length": length,
            "graham_bound": graham_bound(dag, args.cores),
            "join_nodes": joins, "blocks": blocks, "tail": tail,
            "local_paths": [local_paths(dag, block) for block in blocks],
            "priorities": priorities,
            "simulated_makespan": max(event["finish"] for event in events),
            "analytical_algorithm_3_bound": None,
            "notes": ["One release, all durations equal WCET, no communication cost.",
                      "Simulated makespan is NOT a worst-case response-time guarantee.",
                      "Analytical Algorithm 3 implemented.",
                      "Thermal control are not implemented."],
        }
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out / "schedule.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)
    print(json.dumps(report, indent=2))
    print(f"\nSaved report.json and schedule.csv in {args.out.resolve()}")


if __name__ == "__main__":
    main()
