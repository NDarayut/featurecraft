import numpy as np

from featurecraft.operators import OPERATORS, select_operators


def test_nan_domains_never_inf():
    x = np.array([-5.0, -1.0, 0.0, 4.0, np.nan])
    for name in ("log1p", "sqrt", "reciprocal", "square", "abs"):
        out = OPERATORS[name].fn(x)
        assert not np.isinf(out).any(), name
    assert np.isnan(OPERATORS["log1p"].fn(np.array([-2.0]))[0])
    assert np.isnan(OPERATORS["sqrt"].fn(np.array([-1.0]))[0])
    assert np.isnan(OPERATORS["reciprocal"].fn(np.array([0.0]))[0])


def test_div_by_zero_is_nan():
    out = OPERATORS["div"].fn(np.array([1.0, 2.0]), np.array([0.0, 4.0]))
    assert np.isnan(out[0]) and out[1] == 0.5


def test_freq_state_and_unseen():
    op = OPERATORS["freq"]
    train = np.array(["a", "a", "b", np.nan], dtype=object)
    state = op.fit_state(train)
    vals = op.apply(state, np.array(["a", "b", "zzz", np.nan], dtype=object))
    assert vals.tolist() == [2.0, 1.0, 0.0, 1.0]  # unseen -> 0, NaN its own level


def test_groupby_mean_fallback():
    op = OPERATORS["groupby_mean"]
    cat = np.array(["a"] * 6 + ["b"] * 6 + ["tiny"], dtype=object)
    num = np.array([1.0] * 6 + [3.0] * 6 + [100.0])
    state = op.fit_state(num, cat)
    out = op.apply(state, num, np.array(["a", "b", "tiny", "unseen"], dtype=object))
    assert out[0] == 1.0 and out[1] == 3.0
    # tiny group (< MIN_GROUP_SIZE) and unseen both fall back to the global mean
    assert out[2] == out[3] == state["fallback"]


def test_cat_cross_unseen_pair():
    op = OPERATORS["cat_cross"]
    a = np.array(["x", "x", "y"], dtype=object)
    b = np.array(["1", "2", "1"], dtype=object)
    state = op.fit_state(a, b)
    out = op.apply(state, np.array(["x", "y"], dtype=object), np.array(["2", "2"], dtype=object))
    assert out[0] >= 0 and out[1] == -1  # (y,2) never seen in train


def test_select_operators_subset_and_unknown():
    subset = select_operators(["add", "mul"])
    assert set(subset) == {"add", "mul"}
    try:
        select_operators(["nope"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
