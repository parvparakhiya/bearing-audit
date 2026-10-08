import pytest

from bearing_audit import bearing


@pytest.mark.parametrize("name", sorted(bearing.CWRU_PUBLISHED))
def test_geometry_reproduces_cwru_published_multiples(name):
    assert bearing.SKF_6205.multiples()[name] == pytest.approx(bearing.CWRU_PUBLISHED[name], rel=1e-3)


def test_ball_impact_rate_is_twice_the_spin_frequency():
    m = bearing.SKF_6205.multiples()
    assert m["2xBSF"] == pytest.approx(2 * m["BSF"])
    assert m["BSF"] == pytest.approx(2.3567, rel=1e-3)  # the textbook value, not CWRU's


def test_frequencies_scale_linearly_with_shaft_speed():
    f1 = bearing.SKF_6205.frequencies(10.0)
    f2 = bearing.SKF_6205.frequencies(30.0)
    for k in f1:
        assert f2[k] == pytest.approx(3 * f1[k])


def test_inner_race_faster_than_outer_race():
    m = bearing.SKF_6205.multiples()
    assert m["BPFI"] + m["BPFO"] == pytest.approx(bearing.SKF_6205.n_elements)
