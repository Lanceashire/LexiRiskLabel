"""Regression tests for fail-closed parity and direction-aware recovery."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'experiments/ccfb_extension' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


parity = module('analyze_exposure_confirmation')
analysis = module('analyze_dasfaa_revision')


def fixture():
    frame = pd.DataFrame({key:[1.0,2.0] for key in parity.NUMERIC_FIELDS})
    frame['dataset'], frame['method'], frame['seed'], frame['rho'], frame['round'] = 'test','policy',1,.1,[0,1]
    for col in parity.EXACT_FIELDS:
        frame[col] = 'abc'
    return frame


def test_parity_missing_field_fails_closed():
    a = fixture()
    with pytest.raises(ValueError, match='missing required parity fields'):
        parity.compare_frames(a.drop(columns='brier'), a)


def test_parity_actual_predictive_metric_mismatch():
    a, b = fixture(), fixture()
    a.loc[0,'recall'] += .001
    assert not parity.compare_frames(a,b)['parity_passed']


def test_parity_matching_nan_allowed_but_one_sided_not():
    a, b = fixture(), fixture()
    a.loc[0,'calibration_slope'] = np.nan
    assert not parity.compare_frames(a,b)['parity_passed']
    b.loc[0,'calibration_slope'] = np.nan
    assert parity.compare_frames(a,b)['parity_passed']


def test_parity_infinity_rejected():
    a = fixture()
    a.loc[0,'brier'] = np.inf
    with pytest.raises(ValueError, match='infinity'):
        parity.compare_frames(a,a)


def test_parity_key_permutation_allowed():
    a = fixture()
    assert parity.compare_frames(a.iloc[::-1],a)['parity_passed']


def test_parity_missing_or_duplicate_keys_rejected():
    a = fixture()
    with pytest.raises(ValueError, match='row keys differ'):
        parity.compare_frames(a.iloc[:1],a)
    with pytest.raises(ValueError, match='duplicate row keys'):
        parity.compare_frames(pd.concat([a,a]),a)


def test_nonpositive_oracle_gap_unavailable_raw_gain_retained():
    gain,gap,ratio,flag = analysis.recovery([.5,.5,.5], [.4,.4,.4], [.3,.4,.6])
    np.testing.assert_allclose(gain,[.1,.1,.1])
    assert flag.tolist() == ['NEGATIVE_GAP','ZERO_GAP','POSITIVE_GAP']
    assert np.isnan(ratio[:2]).all()
    assert ratio[2] == pytest.approx(.5)


def test_tiny_positive_and_nonfinite_gap_unavailable():
    _,_,ratio,flag = analysis.recovery([.1,.1], [0,0], [1e-12,np.inf])
    assert flag.tolist() == ['TINY_POSITIVE_GAP','NONFINITE']
    assert np.isnan(ratio).all()


def test_brier_improvement_direction():
    gain,gap,ratio,flag = analysis.recovery([.2,.2],[.3,.3],[.1,.4],higher_better=False)
    np.testing.assert_allclose(gain,[.1,.1])
    np.testing.assert_allclose(gap,[.2,-.1])
    assert ratio[0] == pytest.approx(.5)
    assert flag[1] == 'NEGATIVE_GAP'


def test_cluster_bootstrap_preserves_seed_blocks_and_sign():
    rows=[]
    for seed in range(4):
        for rho in [.1,.2]:
            for method in ['a','b']:
                x=seed + (method=='b') + rho
                rows.append(dict(seed=seed,rho=rho,method=method,x=x,y=-x))
    g=pd.DataFrame(rows)
    for centered in [False,True]:
        r=analysis.cluster_association(g,'x','y',centered=centered,n_boot=100)
        assert r['spearman_rho'] == pytest.approx(-1)
        assert r['n_cells'] == 16 and r['n_seeds'] == 4
        assert r['ci_high'] == pytest.approx(-1)
