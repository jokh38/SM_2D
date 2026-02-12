#!/usr/bin/env python3
"""
Regression check for angular variance mitigation at coarse theta resolution.

Pass criteria:
- Case C (N_theta=36) is no longer variance-collapsed and stays near-unity.
- Case D (N_theta=360) stays near-unity.
- Coarse-vs-fine mismatch |R_theta(D)-R_theta(C)| stays bounded.

Usage:
  python3 docs/val_test/check_theta_resolution_regression.py
  python3 docs/val_test/check_theta_resolution_regression.py --run
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "results" / "one_step_summary" / "metrics.csv"
CASE_C_INI = ROOT / "docs" / "val_test" / "cases" / "case_c.ini"
CASE_D_INI = ROOT / "docs" / "val_test" / "cases" / "case_d.ini"
CASE_C_DIR = ROOT / "results" / "one_step_case_C"
CASE_D_DIR = ROOT / "results" / "one_step_case_D"
DEBUG_EXPORTS = [
    "debug_ps_raw_iter_00_initial.csv",
    "debug_ps_raw_iter_01_after_K4.csv",
    "debug_cells_iter_01_after_K4.csv",
    "debug_channels_totals_iter01.csv",
]


def run_cmd(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_case_and_capture(case_label: str, case_ini: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.log"
    cmd = ["./run_simulation", str(case_ini)]
    print("+", " ".join(cmd), f"(log: {run_log})")
    with run_log.open("w") as logf:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=logf, stderr=subprocess.STDOUT)

    for name in DEBUG_EXPORTS:
        src = ROOT / "results" / name
        if not src.exists():
            raise FileNotFoundError(f"{case_label}: missing debug export '{src}' after run")
        shutil.copy2(src, out_dir / name)


def load_metrics(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")
    rows: Dict[str, Dict[str, str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["case"]] = row
    return rows


def as_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception as exc:
        raise ValueError(f"failed to parse {key}='{row.get(key)}'") from exc


def as_int(row: Dict[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except Exception as exc:
        raise ValueError(f"failed to parse {key}='{row.get(key)}'") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="Rerun case C and D + analysis before checking metrics.",
    )
    parser.add_argument("--min-r-c", type=float, default=0.70, help="Min required R_theta for case C.")
    parser.add_argument("--max-r-c", type=float, default=1.60, help="Max allowed R_theta for case C.")
    parser.add_argument("--min-r-d", type=float, default=0.80, help="Min required R_theta for case D.")
    parser.add_argument("--max-r-d", type=float, default=1.60, help="Max allowed R_theta for case D.")
    parser.add_argument(
        "--max-abs-delta",
        type=float,
        default=0.50,
        help="Max allowed |R_theta(D)-R_theta(C)|.",
    )
    args = parser.parse_args()

    if args.run:
        run_case_and_capture("C", CASE_C_INI, CASE_C_DIR)
        run_case_and_capture("D", CASE_D_INI, CASE_D_DIR)
        run_cmd(["python3", "docs/val_test/analyze_one_step.py"])

    rows = load_metrics(METRICS_PATH)
    if "C" not in rows or "D" not in rows:
        print("FAIL: metrics.csv missing Case C or Case D rows.")
        return 2

    c = rows["C"]
    d = rows["D"]

    n_theta_c = as_int(c, "n_theta")
    n_theta_d = as_int(d, "n_theta")
    r_c = as_float(c, "r_theta")
    r_d = as_float(d, "r_theta")
    delta = r_d - r_c
    abs_delta = abs(delta)

    print(f"Case C: n_theta={n_theta_c}, R_theta={r_c:.6f}")
    print(f"Case D: n_theta={n_theta_d}, R_theta={r_d:.6f}")
    print(f"Delta : R_theta(D)-R_theta(C)={delta:.6f}")

    failures = []
    if n_theta_c != 36:
        failures.append(f"Case C expected n_theta=36, got {n_theta_c}")
    if n_theta_d != 360:
        failures.append(f"Case D expected n_theta=360, got {n_theta_d}")
    if r_c < args.min_r_c:
        failures.append(f"Case C R_theta={r_c:.6f} < min-r-c={args.min_r_c:.6f}")
    if r_c > args.max_r_c:
        failures.append(f"Case C R_theta={r_c:.6f} > max-r-c={args.max_r_c:.6f}")
    if r_d < args.min_r_d:
        failures.append(f"Case D R_theta={r_d:.6f} < min-r-d={args.min_r_d:.6f}")
    if r_d > args.max_r_d:
        failures.append(f"Case D R_theta={r_d:.6f} > max-r-d={args.max_r_d:.6f}")
    if abs_delta > args.max_abs_delta:
        failures.append(f"|Delta|={abs_delta:.6f} > max-abs-delta={args.max_abs_delta:.6f}")

    if failures:
        print("FAIL:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("PASS: coarse and fine angular variance remain in expected near-unity bands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
