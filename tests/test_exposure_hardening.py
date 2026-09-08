from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments" / "ccfb_extension" / "run_ccfb_extension.py"
SPEC = importlib.util.spec_from_file_location("ccfb_extension_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def test_registered_study_inputs_have_strictly_positive_exposure() -> None:
    missing = [
        str(RUNNER.DATASETS[dataset]["path"])
        for dataset in ("uci", "hmeq", "ppdai", "lendingclub")
        if not RUNNER.DATASETS[dataset]["path"].is_file()
    ]
    if missing:
        pytest.skip(
            "Registered raw datasets are intentionally not distributed; "
            "materialize them before running the data-level exposure audit."
        )
    for dataset in ("uci", "hmeq", "ppdai", "lendingclub"):
        _, _, _, exposure, _ = RUNNER.dataset_frame(dataset)
        assert np.isfinite(exposure).all()
        assert (exposure > 0).all()


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([1.0, np.nan]),
        pd.Series([1.0, "not-a-number"]),
        pd.Series([1.0, np.inf]),
        pd.Series([1.0, -1.0]),
        pd.Series([1.0, 0.0]),
    ],
)
def test_invalid_exposure_fails_closed(values: pd.Series) -> None:
    with pytest.raises(ValueError, match="Exposure must be finite and strictly positive"):
        RUNNER.validate_exposure(values, dataset="synthetic", column="exposure")


def test_valid_exposure_is_not_changed() -> None:
    values = pd.Series([1.0, 2.5, 100.0])
    observed = RUNNER.validate_exposure(values, dataset="synthetic", column="exposure")
    np.testing.assert_array_equal(observed, values.to_numpy(float))
