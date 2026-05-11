import numpy as np
import pandas as pd

from monitoring.drift import population_stability_index, kolmogorov_smirnov_statistic


def test_psi_zero_for_identical():
    s1 = pd.Series([1, 2, 3, 4, 5])
    s2 = pd.Series([1, 2, 3, 4, 5])
    psi = population_stability_index(s1, s2)
    assert abs(psi) < 1e-8


def test_psi_detects_shift():
    ref = pd.Series(np.random.normal(0, 1, size=2000))
    cur = pd.Series(np.random.normal(3, 1, size=2000))
    psi = population_stability_index(ref, cur)
    assert psi > 0.2


def test_ks_numeric():
    ref = pd.Series(np.random.normal(0, 1, size=200))
    cur = pd.Series(np.random.normal(0.5, 1, size=200))
    ks = kolmogorov_smirnov_statistic(ref, cur)
    assert 0.0 <= ks <= 1.0
