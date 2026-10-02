import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_sv_split_attempts import summarize_attempts
from diagnose_sv_refit import _resolve_attempt


def _write(root, commodity, variant, attempt, converged, rhat=1.002, ess=900, div=0):
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "commodity": commodity,
        "variant": variant,
        "block": 0,
        "requested_attempt": attempt,
        "converged": converged,
        "fit_seed": 42 + attempt - 1,
        "max_rhat": rhat,
        "min_ess": ess,
        "n_divergences": div,
        "divergence_geometry": {"available": True},
        "error": None,
    }
    (root / f"sv_refit__{commodity}__{variant}__attempt-{attempt}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_distinguishes_statistical_failure_from_skipped_attempt(tmp_path):
    _write(tmp_path, "cocoa", "SV-t", 1, False, div=2)
    _write(tmp_path, "cocoa", "SV-t", 2, True)
    _write(tmp_path, "gold", "SV-t", 1, True)

    status = {
        "cocoa_SV_t_attempt1": "success",
        "cocoa_SV_t_attempt2": "success",
        "gold_SV_t_attempt1": "success",
        "gold_SV_t_attempt2": "skipped",
    }
    result = summarize_attempts(
        tmp_path, [("cocoa", "SV-t", 0), ("gold", "SV-t", 0)], status
    )

    assert result["all_converged"]
    assert result["infrastructure_complete"]
    cocoa = result["cases"][0]
    assert cocoa["attempts"][0]["execution_status"] == "statistical_failure"
    assert cocoa["accepted_attempt"] == 2
    gold = result["cases"][1]
    assert gold["attempts"][1]["execution_status"] == "skipped"


def test_missing_artifact_after_cancel_is_infrastructure_incomplete(tmp_path):
    _write(tmp_path, "cocoa", "SV-t", 1, False, div=1)
    _write(tmp_path, "gold", "SV-t", 1, True)

    status = {
        "cocoa_SV_t_attempt1": "success",
        "cocoa_SV_t_attempt2": "cancelled",
        "gold_SV_t_attempt1": "success",
        "gold_SV_t_attempt2": "skipped",
    }
    result = summarize_attempts(
        tmp_path, [("cocoa", "SV-t", 0), ("gold", "SV-t", 0)], status
    )

    assert not result["infrastructure_complete"]
    cocoa = result["cases"][0]
    assert cocoa["attempts"][1]["execution_status"] == "infrastructure_incomplete"


def test_split_attempt_seed_and_budget_match_production():
    a1, seed1, requested1 = _resolve_attempt(1, 42)
    a2, seed2, requested2 = _resolve_attempt(2, 42)

    assert requested1 == 1
    assert requested2 == 2
    assert seed1 == 42
    assert seed2 == 43
    assert a1 == [{"chains": 4, "tune": 1000, "draws": 1000}]
    assert a2 == [{"chains": 4, "tune": 2000, "draws": 2000}]
