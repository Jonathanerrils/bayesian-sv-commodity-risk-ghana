import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import production_runner as runner


class ProductionRunnerWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.checkpoint_dir = Path(self.tmpdir.name)
        self.checkpoint_patch = patch.object(
            runner, "CHECKPOINT_DIR", self.checkpoint_dir
        )
        self.checkpoint_patch.start()

        idx = pd.date_range("2024-01-01", periods=12, freq="D")
        self.returns = pd.Series(range(12), index=idx, dtype=float)
        self.prices = pd.Series(range(100, 112), index=idx, dtype=float)
        self.logger = logging.getLogger("production-runner-tests")

    def tearDown(self):
        self.checkpoint_patch.stop()
        self.tmpdir.cleanup()

    @staticmethod
    def forecast_frame():
        return pd.DataFrame(
            {
                "var_0.01": [0.1],
                "es_0.01": [0.2],
                "var_0.05": [0.05],
                "es_0.05": [0.1],
            },
            index=pd.DatetimeIndex(["2024-01-12"], name="date"),
        )

    def test_checkpoint_paths_isolate_window_and_sv_refit_config(self):
        w750 = runner.checkpoint_path("gold", "GARCH", 750)
        w1000 = runner.checkpoint_path("gold", "GARCH", 1000)
        sv42 = runner.checkpoint_path("gold", "SV-t", 750, 42)
        sv21 = runner.checkpoint_path("gold", "SV-t", 750, 21)

        self.assertNotEqual(w750, w1000)
        self.assertNotEqual(sv42, sv21)
        self.assertIn("w750", w750.name)
        self.assertIn("refit42", sv42.name)

    def test_nondefault_window_never_uses_legacy_checkpoint(self):
        legacy = runner.legacy_checkpoint_path("gold", "GARCH")
        legacy.write_text("date,var_0.01\n2024-01-01,0.1\n", encoding="utf-8")

        self.assertIsNone(
            runner.existing_checkpoint_path("gold", "GARCH", 750)
        )
        self.assertEqual(
            runner.existing_checkpoint_path("gold", "GARCH", runner.WINDOW),
            legacy,
        )

    def test_window_is_forwarded_to_all_benchmark_models(self):
        window = 7
        cases = [
            ("GARCH", "rolling_var_es"),
            ("EGARCH", "rolling_var_es"),
            ("HistSim", "historical_simulation_var_es"),
            ("OU", "rolling_ou_var_es"),
        ]

        for model, target_name in cases:
            with self.subTest(model=model):
                target = f"production_runner.{target_name}"
                with patch(target, return_value=self.forecast_frame()) as mocked:
                    runner.run_benchmark(
                        "gold",
                        model,
                        self.returns,
                        self.prices,
                        self.logger,
                        window=window,
                    )

                self.assertEqual(mocked.call_args.kwargs["window"], window)

    def test_window_and_refit_are_forwarded_to_sv_runner(self):
        window = 7
        refit_every = 21
        with patch(
            "production_runner.rolling_sv_var_es",
            return_value=self.forecast_frame(),
        ) as mocked:
            runner.run_sv(
                "gold",
                "SV-t",
                self.returns,
                self.logger,
                window=window,
                refit_every=refit_every,
            )

        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["window"], window)
        self.assertEqual(kwargs["refit_every"], refit_every)
        self.assertIn("w7", kwargs["checkpoint_path"].name)
        self.assertIn("refit21", kwargs["checkpoint_path"].name)


if __name__ == "__main__":
    unittest.main()
