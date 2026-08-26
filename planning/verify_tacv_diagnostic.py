#!/usr/bin/env python3
"""Independent raw-record reconciliation of the TACV diagnostic gates."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "artifacts/transient_aware_candidate_veto/20260824_184551"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    output = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        output[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return output


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    return float(np.corrcoef(ranks(first), ranks(second))[0, 1])


def main() -> None:
    checks: dict[str, bool] = {}
    reproduction_counts: dict[str, int] = {}
    for block in ("development", "holdout"):
        directory = ROOT / f"02_predictability/diagnostic_runs/{block}/episode_records"
        paths = sorted(directory.glob("*_reproduction.json"))
        reproduction_counts[block] = len(paths)
        checks[f"{block}_reproduction_count_100"] = len(paths) == 100
        checks[f"{block}_all_reproductions_pass"] = all(
            load_json(path)["status"] == "PASS" for path in paths
        )
        checks[f"{block}_candidate_preview_files_100"] = len(
            list(directory.glob("*_candidate_previews.jsonl.gz"))
        ) == 100
        gzip_valid = True
        for path in directory.glob("*_candidate_previews.jsonl.gz"):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    next(handle)
            except Exception:
                gzip_valid = False
                break
        checks[f"{block}_candidate_preview_gzip_valid"] = gzip_valid

    with (ROOT / "02_predictability/PREVIEW_VS_REALIZED_TRANSIENT.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        transient = list(csv.DictReader(handle))
    predictability = load_json(
        ROOT / "02_predictability/PREVIEW_TRANSIENT_PREDICTABILITY.json"
    )
    rho_recomputed: dict[str, float] = {}
    for block in ("development", "holdout"):
        selected = [
            row
            for row in transient
            if row["block"] == block and row["clean_window_H4"].lower() == "true"
        ]
        rho = spearman(
            [float(row["J_preview"]) for row in selected],
            [float(row["J_real_mean_H0p4"]) for row in selected],
        )
        expected = predictability["sections"][f"{block}_clean_window"]["spearman_rho"]
        rho_recomputed[block] = rho
        checks[f"{block}_clean_rho_reproduced"] = abs(rho - expected) < 1.0e-12
        checks[f"{block}_clean_count_reproduced"] = len(selected) == int(
            predictability["sections"][f"{block}_clean_window"]["event_count"]
        )

    with (ROOT / "03_replaceability/NECESSARY_VS_AVOIDABLE_TRANSIENTS.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        events = list(csv.DictReader(handle))
    replaceability = load_json(
        ROOT / "03_replaceability/SAFE_ALTERNATIVE_REPLACEABILITY.json"
    )
    rates: dict[str, float] = {}
    for block in ("development", "holdout"):
        selected = [row for row in events if row["block"] == block]
        rate = float(
            np.mean([row["replaceable"].lower() == "true" for row in selected])
        )
        rates[block] = rate
        checks[f"{block}_replaceability_rate_reproduced"] = abs(
            rate - float(replaceability["blocks"][block]["replaceable_rate"])
        ) < 1.0e-12
        checks[f"{block}_high_event_count_reproduced"] = len(selected) == int(
            replaceability["blocks"][block]["high_transient_event_count"]
        )

    gate = load_json(ROOT / "04_gate_decision/TACV_GATE_DECISION.json")
    checks["gate_predictability_matches"] = (
        gate["PREVIEW_TRANSIENT_PREDICTABILITY"]
        == predictability["PREVIEW_TRANSIENT_PREDICTABILITY"]
    )
    checks["gate_replaceability_matches"] = (
        gate["SAFE_ALTERNATIVE_HEADROOM"]
        == replaceability["SAFE_ALTERNATIVE_HEADROOM"]
    )
    checks["gate_authorization_consistent"] = (
        gate["TACV_AUTHORIZED"] == "YES"
        and predictability["TACV_PREDICTABILITY_GATE_PASS"]
        and replaceability["TACV_REPLACEABILITY_GATE_PASS"]
    )
    payload = {
        "schema_version": "tacv_diagnostic_independent_reconciliation_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reproduction_counts": reproduction_counts,
        "clean_spearman_recomputed": rho_recomputed,
        "replaceable_rates_recomputed": rates,
        "formal_v2_read_for_parameter_selection": False,
    }
    target = ROOT / "00_context/DIAGNOSTIC_INDEPENDENT_RECONCILIATION.json"
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
