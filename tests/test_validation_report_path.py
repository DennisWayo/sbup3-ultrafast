from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


class TestValidationReportPath(unittest.TestCase):
    def test_unique_report_path_suffixes_on_collision(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        validate_mod = _load_module(
            repo / "analysis" / "validate_sbup3.py",
            "validate_sbup3_path_test_mod",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            ts = "20260520_000000_000000"
            (report_dir / f"validation_{ts}.json").write_text("{}\n")
            (report_dir / f"validation_{ts}_01.json").write_text("{}\n")

            path = validate_mod._unique_validation_report_path(report_dir, ts)
            self.assertEqual(path.name, f"validation_{ts}_02.json")


if __name__ == "__main__":
    unittest.main()

