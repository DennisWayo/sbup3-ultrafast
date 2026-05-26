from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


class TestPipelineCalibrationPreflight(unittest.TestCase):
    def test_missing_calibration_channel_fails_before_pipeline_run(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        run_pipeline = repo / "analysis" / "run_pipeline.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_csv = Path(tmpdir) / "ref.csv"
            ref_csv.write_text(
                "\n".join(
                    [
                        "energy_eV,alpha_m_inv,n",
                        "1.5,1000,3.5",
                        "2.0,2000,3.7",
                    ]
                )
                + "\n"
            )

            proc = subprocess.run(
                [
                    str(repo / ".venv" / "bin" / "python"),
                    str(run_pipeline),
                    "--mode",
                    "sequential",
                    "--calibrate-scale",
                    "--calibration-ref-csv",
                    str(ref_csv),
                    "--calibration-fit-channels",
                    "chi_imag",
                ],
                cwd=str(repo),
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(proc.returncode, 0)
            combined = f"{proc.stdout}\n{proc.stderr}".lower()
            self.assertIn("missing required channels", combined)
            self.assertIn("requested: chi_imag", combined)


if __name__ == "__main__":
    unittest.main()

