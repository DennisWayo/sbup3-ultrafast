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


class TestReferenceCsvAliases(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path(__file__).resolve().parents[1]
        self.validate_mod = _load_module(
            self.repo / "analysis" / "validate_sbup3.py",
            "validate_sbup3_test_mod",
        )
        self.calibrate_mod = _load_module(
            self.repo / "analysis" / "calibrate_polarization_scale.py",
            "calibrate_polarization_scale_test_mod",
        )

    def test_alpha_m_inv_alias_is_loaded_as_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ref.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "energy_eV,alpha_m_inv,n",
                        "1.5,1000,3.5",
                        "2.0,2000,3.7",
                    ]
                )
                + "\n"
            )

            val_ref = self.validate_mod._load_reference_csv(csv_path)
            cal_ref = self.calibrate_mod._load_reference_csv(csv_path)

            self.assertIn("alpha", val_ref)
            self.assertIn("alpha", cal_ref)
            self.assertEqual(val_ref["alpha"].tolist(), [1000.0, 2000.0])
            self.assertEqual(cal_ref["alpha"].tolist(), [1000.0, 2000.0])


if __name__ == "__main__":
    unittest.main()

