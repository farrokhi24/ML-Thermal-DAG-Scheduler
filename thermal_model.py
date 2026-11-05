"""Synthetic per-core thermal data driven by a DAG schedule.

Exact RC integration between power-switching events; optional spatial coupling.
This is an explicitly uncalibrated data source for the thermal research phase,
not a reproduction of the thermal paper's 8x8 NoC traffic simulator.
Power columns are instantaneous, right-continuous values at the row timestamp.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd


def _positive(name, value, zero=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or (
            value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def grid_topology(cores, columns=None):
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise ValueError("cores must be a positive integer")
    columns = math.ceil(math.sqrt(cores)) if columns is None else columns
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer")
    positions = {c: (c // columns, c % columns) for c in range(cores)}
    neighbors = {c: [d for d in range(cores) if c != d and
                     max(abs(positions[c][0] - positions[d][0]),
                         abs(positions[c][1] - positions[d][1])) == 1]
                 for c in range(cores)}
    return positions, neighbors, columns


class RCNetwork:
    """Stateful exact RC plant shared by trace generation and closed-loop runs."""
    def __init__(self, cores, *, ambient=45.0, resistance=.8, capacitance=3.0,
                 coupling=0.0, columns=None, initial_temperature=None):
        _positive('resistance', resistance)
        _positive('capacitance', capacitance)
        _positive('coupling', coupling, zero=True)
        initial_temperature = ambient if initial_temperature is None else initial_temperature
        if not all(math.isfinite(v) for v in [ambient, initial_temperature]):
            raise ValueError('Temperatures must be finite')
        self.positions, self.neighbors, self.columns = grid_topology(cores, columns)
        self.cores, self.ambient = cores, ambient
        conductance = np.eye(cores) / resistance
        for c in range(cores):
            conductance[c, c] += coupling * len(self.neighbors[c])
            for d in self.neighbors[c]:
                conductance[c, d] = -coupling
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(conductance / capacitance)
        if not np.isfinite(self.eigenvalues).all() or np.any(self.eigenvalues <= 0):
            raise ValueError('RC parameters are numerically ill-conditioned')
        self.inverse = (self.eigenvectors / (self.eigenvalues * capacitance)) @ self.eigenvectors.T
        self.state = np.full(cores, initial_temperature - ambient, dtype=float)

    @property
    def temperatures(self):
        return self.state.copy() + self.ambient

    def advance(self, power, seconds):
        _positive('elapsed seconds', seconds, zero=True)
        power = np.asarray(power, dtype=float)
        if power.shape != (self.cores,) or not np.isfinite(power).all() or np.any(power < 0):
            raise ValueError('Expected one finite nonnegative power per core')
        equilibrium = self.inverse @ power
        self.state = equilibrium + self.eigenvectors @ (
            np.exp(-self.eigenvalues * seconds) * (self.eigenvectors.T @ (self.state - equilibrium)))
        if not np.isfinite(self.state).all():
            raise ValueError('RC state overflow')
        return self.temperatures


def load_schedule(path, cores=None):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"node", "core", "start", "finish"} <= set(reader.fieldnames or []):
            raise ValueError("Schedule requires node, core, start, finish columns")
        rows = []
        for raw in reader:
            row = {"node": int(raw["node"]), "core": int(raw["core"]),
                   "start": float(raw["start"]), "finish": float(raw["finish"])}
            if row["core"] < 0 or not all(math.isfinite(row[k]) for k in ["start", "finish"]) or not (
                    0 <= row["start"] < row["finish"]):
                raise ValueError("Invalid schedule core or execution interval")
            if "duration" in raw and not math.isclose(float(raw["duration"]),
                                                     row["finish"] - row["start"], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Schedule duration does not match finish-start")
            rows.append(row)
    if not rows:
        raise ValueError("Schedule must contain at least one task")
    if len({r["node"] for r in rows}) != len(rows):
        raise ValueError("Expected one execution per DAG node")
    count = max(r["core"] for r in rows) + 1 if cores is None else cores
    grid_topology(count)
    if any(r["core"] >= count for r in rows):
        raise ValueError("Schedule references a core outside the configured platform")
    for c in range(count):
        tasks = sorted((r for r in rows if r["core"] == c), key=lambda r: r["start"])
        if any(a["finish"] > b["start"] for a, b in zip(tasks, tasks[1:])):
            raise ValueError("Tasks overlap on the same core")
    return sorted(rows, key=lambda r: (r["start"], r["core"], r["node"])), count


def _write_learning_data(frame, path, positions, columns):
    """One-step examples, with explicit horizon and masked 3x3 neighborhoods.

    Each row's inputs use only its current timestamp. The target is the next
    sensor reading. Throttling is zero for this uncontrolled baseline; a future
    controller must produce actual action variation before learning its effect.
    """
    offsets = [(0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0), (1, -1), (0, -1), (-1, -1)]
    at = {pos: core for core, pos in positions.items()}
    fields = ["time", "core", "target_time", "horizon_seconds", "target_temperature"]
    for slot in range(9):
        fields += [f"slot_{slot}_{name}" for name in ["mask", "temperature", "power", "active", "throttled"]]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(frame) - 1):
            current, target = frame.iloc[index], frame.iloc[index + 1]
            for core, (r, c) in positions.items():
                example = {"time": current["Time"], "core": core,
                           "target_time": target["Time"],
                           "horizon_seconds": target["Time_Seconds"] - current["Time_Seconds"],
                           "target_temperature": target[f"Temp_Core_{core}"]}
                for slot, (dr, dc) in enumerate(offsets):
                    neighbor = at.get((r + dr, c + dc))
                    example[f"slot_{slot}_mask"] = int(neighbor is not None)
                    for name, prefix in [("temperature", "Temp"), ("power", "Power"),
                                         ("active", "Active"), ("throttled", "Throttled")]:
                        example[f"slot_{slot}_{name}"] = (
                            current[f"{prefix}_Core_{neighbor}"] if neighbor is not None else 0)
                writer.writerow(example)


def generate_thermal_trace(schedule_csv_path, output_csv_path,
                           T_amb=45.0, R=0.8, C=3.0, dt=1.0, P_idle=2.0, seed=42,
                           *, seconds_per_unit=1.0, cores=None, columns=None,
                           coupling=0.0, initial_temperature=None, power_min=10.0,
                           power_max=20.0, node_powers=None, sensor_noise_std=0.0,
                           cooldown=0.0, plot_path=None, show=False):
    """Export temperatures, power, node-power mapping, features and metadata.

    R: kelvin/watt; C: joule/kelvin; coupling: watt/kelvin per neighbor pair.
    dt and cooldown use schedule units; seconds_per_unit maps them to seconds.
    Latent_Temp is the model's state; Temp includes optional sensor noise.
    Initial state persists through task boundaries and idle/cooldown intervals.
    """
    for name, value in [("R", R), ("C", C), ("dt", dt), ("seconds_per_unit", seconds_per_unit)]:
        _positive(name, value)
    for name, value in [("P_idle", P_idle), ("coupling", coupling), ("power_min", power_min),
                        ("power_max", power_max), ("sensor_noise_std", sensor_noise_std), ("cooldown", cooldown)]:
        _positive(name, value, zero=True)
    if power_min > power_max:
        raise ValueError("power_min must not exceed power_max")
    initial_temperature = T_amb if initial_temperature is None else initial_temperature
    if not all(math.isfinite(t) for t in [T_amb, initial_temperature]):
        raise ValueError("Temperatures must be finite")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    schedule, count = load_schedule(schedule_csv_path, cores)
    positions, neighbors, columns = grid_topology(count, columns)
    rng = random.Random(seed)
    ids = sorted(r["node"] for r in schedule)
    if node_powers is None:
        powers_by_node = {v: rng.uniform(power_min, power_max) for v in ids}
    else:
        powers_by_node = {int(v): float(p) for v, p in node_powers.items()}
        if set(powers_by_node) != set(ids):
            raise ValueError("Exactly one power is required per scheduled node")
        for power in powers_by_node.values():
            _positive("node power", power, zero=True)
    horizon = max(r["finish"] for r in schedule) + cooldown
    if not math.isfinite(horizon) or not math.isfinite(horizon * seconds_per_unit):
        raise ValueError("Thermal time scale overflow")
    if not math.isfinite(horizon / dt):
        raise ValueError("Thermal trace too large; increase dt")
    steps = math.floor(horizon / dt)
    if steps > 1_000_000 or (steps + 2) * count > 5_000_000:
        raise ValueError("Thermal trace too large; increase dt")
    samples = [i * dt for i in range(steps + 1) if i * dt < horizon]
    samples.append(horizon)  # Preserve a fractional final interval exactly.
    switches = {0.0: []}
    for row in schedule:
        switches.setdefault(row["start"], []).append((row["core"], row["node"]))
        switches.setdefault(row["finish"], []).append((row["core"], None))
    # Process ALL completions before starts at a shared timestamp.
    for changes in switches.values():
        changes.sort(key=lambda item: (item[1] is not None, item[0]))
    boundaries = sorted(set(samples) | set(switches))
    sample_set = set(samples)
    plant = RCNetwork(count, ambient=T_amb, resistance=R, capacitance=C,
                      coupling=coupling, columns=columns, initial_temperature=initial_temperature)
    active = [None] * count
    power = np.full(count, P_idle, dtype=float)
    sensor_rng = random.Random(seed ^ 0x5EED)
    records, previous = [], 0.0
    for time in boundaries:
        elapsed = (time - previous) * seconds_per_unit
        if elapsed > 0:
            plant.advance(power, elapsed)
        for core, node in switches.get(time, []):
            active[core] = node
            power[core] = P_idle if node is None else powers_by_node[node]
        if time in sample_set:
            row = {"Time": time, "Time_Seconds": time * seconds_per_unit}
            for core in range(count):
                latent = float(plant.temperatures[core])
                row[f"Temp_Core_{core}"] = latent + sensor_rng.gauss(0, sensor_noise_std)
                row[f"Latent_Temp_Core_{core}"] = latent
                row[f"Power_Core_{core}"] = float(power[core])
                row[f"Active_Core_{core}"] = int(active[core] is not None)
                row[f"Throttled_Core_{core}"] = 0
            records.append(row)
        previous = time
    frame = pd.DataFrame(records)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Thermal model produced nonfinite values; check physical scales")
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    power_path = out.with_name(out.stem + "_node_power.csv")
    with power_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["node", "core", "start", "finish", "power_watts"])
        writer.writeheader()
        writer.writerows({**r, "power_watts": powers_by_node[r["node"]]} for r in schedule)
    features = out.with_name(out.stem + "_learning.csv")
    _write_learning_data(frame, features, positions, columns)
    metadata = {
        "source": "synthetic, uncalibrated RC network; no hardware measurements",
        "schedule": str(Path(schedule_csv_path).resolve()), "seed": seed, "cores": count,
        "parameters": {"ambient_celsius": T_amb, "initial_celsius": initial_temperature,
                       "resistance_kelvin_per_watt": R, "capacitance_joule_per_kelvin": C,
                       "coupling_watt_per_kelvin": coupling, "idle_watts": P_idle,
                       "power_min_watts": power_min, "power_max_watts": power_max,
                       "sensor_noise_std_celsius": sensor_noise_std, "dt_schedule_units": dt,
                       "seconds_per_schedule_unit": seconds_per_unit, "cooldown_schedule_units": cooldown},
        "topology": {"columns": columns, "positions": positions, "neighbors": neighbors,
                     "policy": "synthetic 2D grid with up to eight neighbors"},
        "integration": "exact linear RC solution, split at every start/finish event",
        "power_semantics": "instantaneous right-continuous power at Time; Temp reflects preceding history",
        "target_semantics": "next sensor sample; inputs use current sample only",
        "feature_slot_order": ["self", "N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        "limitations": ["No thermal controller, DVFS, or migration",
                        "All throttling inputs are zero in this uncontrolled dataset",
                        "Final learning horizon may be shorter than dt; use horizon_seconds or omit that row",
                        "Split train/test chronologically or by DAG; do not randomly mix neighboring samples",
                        "Not the thermal paper's calibrated 8x8 NoC simulator"],
        "files": {"trace": out.name, "node_power": power_path.name, "learning": features.name},
    }
    out.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    if plot_path is not None or show:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for core in range(count):
            ax.plot(frame["Time_Seconds"], frame[f"Temp_Core_{core}"], label=f"Core {core}")
        ax.set(xlabel="Time (seconds)", ylabel="Temperature (°C)", title="Synthetic RC core temperatures")
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
        if plot_path is not None:
            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(plot_path)
        if show:
            plt.show()
        plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=Path("outputs/paper_demo/schedule.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/paper_demo/thermal_trace.csv"))
    parser.add_argument("--seconds-per-unit", type=float, required=True)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--coupling", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--sensor-noise", type=float, default=0.0)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    try:
        frame = generate_thermal_trace(args.schedule, args.out, dt=args.dt, seed=args.seed,
                                       seconds_per_unit=args.seconds_per_unit, cores=args.cores,
                                       columns=args.columns, coupling=args.coupling,
                                       cooldown=args.cooldown, sensor_noise_std=args.sensor_noise,
                                       plot_path=args.plot)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Saved {len(frame)} thermal samples and learning data in {args.out.resolve().parent}")


if __name__ == "__main__":
    main()
