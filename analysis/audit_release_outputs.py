"""Compare current local arrays with a historical SBUP3 run report.

The report is read only. No output array or archived checksum is replaced.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = (ROOT / "archive/2026-10-17-solver-optics/legacy"
          / "run_release_primary_jellison_20260520_081916.json")


def main():
    report = json.loads(REPORT.read_text())
    records = []
    for group, outputs in report["outputs"].items():
        for key, expected in outputs.items():
            location = ROOT / ("sbe" if group == "sbe" else "uppe/outputs")
            path = location / Path(expected["path"]).name
            observed_hash = (hashlib.sha256(path.read_bytes()).hexdigest()
                             if path.is_file() else None)
            records.append({
                "group": group,
                "key": key,
                "path": str(path.relative_to(ROOT)),
                "expected_sha256": expected["sha256"],
                "observed_sha256": observed_hash,
                "state": ("missing" if observed_hash is None else
                          "match" if observed_hash == expected["sha256"] else
                          "mismatch"),
            })
    counts = {status: sum(row["state"] == status for row in records)
              for status in ("match", "mismatch", "missing")}
    print(json.dumps({"historical_report": str(REPORT.relative_to(ROOT)),
                      "historical_report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
                      "counts": counts, "outputs": records}, indent=2))


if __name__ == "__main__":
    main()
