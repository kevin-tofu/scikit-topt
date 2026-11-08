"""Compare end-to-end MMA optimization timings across FEM backends."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tomllib


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("run_case.py")


def run_once(backend: str, args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    env["SKTOPT_FEM_BACKEND"] = backend
    source_paths = [
        ROOT / "scikit-topt",
        ROOT / "packages/mmapy/src",
        ROOT / "packages/scikit-fem-native/python",
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in source_paths]
        + ([existing_pythonpath] if existing_pythonpath else [])
    )
    command = [
        sys.executable,
        str(WORKER),
        "--mesh-size",
        str(args.mesh_size),
        "--iterations",
        str(args.iterations),
        "--record-times",
        str(args.record_times),
        "--volume-fraction",
        str(args.volume_fraction),
        "--move-limit",
        str(args.move_limit),
        "--filter-radius",
        str(args.filter_radius),
        "--solver-option",
        args.solver_option,
    ]
    process_start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "backend": backend,
            "timeout_seconds": args.timeout,
            "process_seconds": time.perf_counter() - process_start,
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
        }

    payload = next(
        (
            line.removeprefix("MMA_BACKEND_RESULT ")
            for line in reversed(completed.stdout.splitlines())
            if line.startswith("MMA_BACKEND_RESULT ")
        ),
        None,
    )
    if payload is None:
        return {
            "status": "failed",
            "backend": backend,
            "return_code": completed.returncode,
            "error": "Worker did not emit a result record.",
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }

    result = json.loads(payload)
    result["return_code"] = completed.returncode
    result["process_seconds"] = time.perf_counter() - process_start
    if completed.stderr:
        result["stderr_tail"] = completed.stderr[-4000:]
    return result


def summarize(results: dict[str, list[dict]]) -> list[str]:
    lines = [
        "backend      ok/repeats   median optimize (s)   median process (s)   median MMA adapter (s)   median compliance   median |volume error|",
        "------------ -----------  -------------------  ------------------  ----------------------  -----------------  ---------------------",
    ]
    for backend, runs in results.items():
        successful = [run for run in runs if run.get("status") == "success"]
        if not successful:
            lines.append(
                f"{backend:<12} 0/{len(runs):<9}  no successful runs"
            )
            failures = [run for run in runs if run.get("status") != "success"]
            if failures:
                failure = failures[0]
                lines.append(
                    f"  first failure: {failure.get('exception_type', failure.get('status'))}: "
                    f"{failure.get('error', failure.get('stderr_tail', ''))}"
                )
            continue
        lines.append(
            f"{backend:<12} {len(successful)}/{len(runs):<9}  "
            f"{statistics.median(r['optimization_seconds'] for r in successful):>19.3f}  "
            f"{statistics.median(r['process_seconds'] for r in successful):>18.3f}  "
            f"{statistics.median(r['mma_adapter_update_seconds'] for r in successful):>22.3f}  "
            f"{statistics.median(r['compliance'] for r in successful):>17.6g}  "
            f"{statistics.median(abs(r['volume_error']) for r in successful):>21.3g}"
        )

    standard = [r for r in results.get("skfem", []) if r.get("status") == "success"]
    native = [r for r in results.get("skfemntv", []) if r.get("status") == "success"]
    if standard and native:
        standard_time = statistics.median(r["optimization_seconds"] for r in standard)
        native_time = statistics.median(r["optimization_seconds"] for r in native)
        lines.append(f"End-to-end speedup (skfem / skfemntv): {standard_time / native_time:.3f}x")
    else:
        lines.append("End-to-end speedup: unavailable until both backends complete successfully.")
    return lines


def native_source_info() -> dict:
    native_root = ROOT / "packages/scikit-fem-native"
    try:
        version = tomllib.loads(
            (native_root / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"]
        commit = subprocess.run(
            ["git", "-C", str(native_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, KeyError, subprocess.CalledProcessError):
        return {"available": False}
    return {"available": True, "version": version, "commit": commit}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["skfem", "skfemntv"])
    parser.add_argument("--mesh-size", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--record-times", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--volume-fraction", type=float, default=0.8)
    parser.add_argument("--move-limit", type=float, default=0.2)
    parser.add_argument("--filter-radius", type=float, default=0.1)
    parser.add_argument(
        "--solver-option",
        choices=("spsolve", "cg_pyamg"),
        default="spsolve",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/mma_backend/results/latest.json"),
    )
    args = parser.parse_args()

    if args.repeats < 1 or args.iterations < 1 or args.record_times < 1:
        parser.error("--repeats, --iterations, and --record-times must be positive")
    unknown = set(args.backends) - {"skfem", "skfemntv"}
    if unknown:
        parser.error(f"unknown backend(s): {', '.join(sorted(unknown))}")

    results = {backend: [] for backend in args.backends}
    for repeat in range(args.repeats):
        order = list(args.backends)
        if repeat % 2:
            order.reverse()
        for backend in order:
            print(f"Run {repeat + 1}/{args.repeats}: {backend}", flush=True)
            result = run_once(backend, args)
            result["repeat"] = repeat + 1
            results[backend].append(result)
            print(f"  status={result['status']}", flush=True)

    report = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "native_backend_source": native_source_info(),
        "configuration": {
            "backends": args.backends,
            "mesh_size": args.mesh_size,
            "iterations": args.iterations,
            "record_times": args.record_times,
            "repeats": args.repeats,
            "volume_fraction": args.volume_fraction,
            "move_limit": args.move_limit,
            "filter_radius": args.filter_radius,
            "solver_option": args.solver_option,
        },
        "results": results,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n".join(summarize(results)))
    print(f"\nFull report: {output}")
    return 0 if all(
        run.get("status") == "success"
        for runs in results.values()
        for run in runs
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
