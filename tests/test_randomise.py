import numpy as np
import pytest

from dynsim import (
    AddAppendage,
    AddGroundDamper,
    AddGroundSpring,
    AppendageSpec,
    GroundDamperSpec,
    GroundSpringSpec,
    ScaleConnection,
    ScaleConnectionSpec,
    ScaleMass,
    ScaleMassSpec,
    build_system,
    make_mdof_chain,
    make_mdof_tower,
    make_state_matrices,
    sample_system,
)


def counts(rs):
    out = {}
    for f in rs.features:
        out[type(f).__name__] = out.get(type(f).__name__, 0) + 1
    return out


# ---- features ------------------------------------------------------------------------
def test_features_change_matrices_as_described():
    base = make_mdof_tower(4)
    s = build_system(
        "tower", 4, features=[ScaleConnection(2, 2.0), AddGroundSpring(3, 100.0, 1.0), ScaleMass(1, 3.0)]
    ).system
    expected = np.zeros((4, 4))
    expected[1:3, 1:3] += 5000 * np.array([[1, -1], [-1, 1]])  # connection 2 joins floors 1 and 2
    expected[3, 3] += 100
    np.testing.assert_allclose(s.K - base.K, expected)
    np.testing.assert_allclose(np.diag(s.M), [5, 15, 5, 5])


def test_appendage_matrices():
    base = make_mdof_tower(4)
    rs = build_system("tower", 4, features=[AddAppendage(3, 2.0, 100.0, 0.5)])
    K, C = rs.system.K, rs.system.C
    assert rs.system.ndof == 5 and rs.ndof == 4
    np.testing.assert_allclose(np.diag(rs.system.M), [5, 5, 5, 5, 2.0])
    np.testing.assert_allclose(K[:4, :4], base.K + np.diag([0, 0, 0, 100.0]))
    assert K[4, 4] == 100 and K[3, 4] == K[4, 3] == -100
    assert C[4, 4] == 0.5 and C[3, 4] == -0.5 and np.isclose(C[3, 3], base.C[3, 3] + 0.5)
    assert "appendage on floor 3" in rs.describe()[0]


def test_appendages_get_consecutive_dofs_and_compose_with_other_features():
    rs = build_system(
        "chain", 3, features=[AddAppendage(0, 1.0, 50.0), AddAppendage(2, 3.0, 70.0), ScaleMass(1, 2.0)]
    )
    assert rs.system.ndof == 5
    np.testing.assert_allclose(np.diag(rs.system.M), [5, 10, 5, 1, 3])
    assert rs.system.K[4, 2] == -70 and rs.system.K[3, 0] == -50


@pytest.mark.parametrize("bad", [AddAppendage(5, 1, 1), AddAppendage(-1, 1, 1)])
def test_appendage_out_of_range_raises(bad):
    with pytest.raises(IndexError):
        build_system("tower", 4, features=[bad])


def test_out_of_range_features_raise():
    with pytest.raises(IndexError):
        build_system("tower", 4, features=[ScaleConnection(4, 2.0)])  # towers have indices 0..3
    with pytest.raises(IndexError):
        build_system("tower", 4, features=[AddGroundSpring(4, 1.0)])
    with pytest.raises(IndexError):
        build_system("tower", 4, features=[ScaleMass(4, 1.0)])
    build_system("chain", 4, features=[ScaleConnection(4, 2.0)])  # chains have one more connection
    with pytest.raises(ValueError):
        build_system("pendulum", 4)


# ---- feature_types ---------------------------------------------------------------------
def test_empty_feature_types_gives_baseline():
    for kind in ("tower", "chain"):
        rs = sample_system(0, kind, 6, feature_types=[])
        assert rs.features == ()


def test_feature_type_selection():
    r = sample_system(1, "tower", 6, feature_types=["mass"])
    assert r.features and all(isinstance(f, ScaleMass) for f in r.features)
    np.testing.assert_allclose(r.system.K, make_mdof_tower(6).K)

    r = sample_system(1, "tower", 6, feature_types=["scale", "ground"])
    assert not any(isinstance(f, ScaleMass) for f in r.features)
    np.testing.assert_allclose(np.diag(r.system.M), 5)


def test_default_feature_types_exclude_appendages():
    assert not any(
        isinstance(f, AddAppendage) for s in range(50) for f in sample_system(s, "tower", 6).features
    )


def test_unknown_or_invalid_feature_types_raise():
    with pytest.raises(ValueError, match="Unknown feature type"):
        sample_system(0, "tower", 3, feature_types=["nope"])
    with pytest.raises(TypeError):
        sample_system(0, "tower", 3, feature_types=[5])


def test_custom_feature_type_object_is_accepted():
    class AlwaysHeavyBase:
        def sample(self, rng, ctx):
            return [ScaleMass(0, 2.0)]

    assert sample_system(0, "tower", 3, feature_types=[AlwaysHeavyBase()]).system.M[0, 0] == 10


# ---- counts: int shorthand and ranges -------------------------------------------------
@pytest.mark.parametrize("seed", range(20))
def test_exact_counts_with_int_and_with_equal_range(seed):
    specs = [ScaleConnectionSpec(n=2), GroundSpringSpec(n=0), ScaleMassSpec(n=(1, 1))]
    got = counts(sample_system(seed, "tower", 8, feature_types=specs))
    assert got == {"ScaleConnection": 2, "ScaleMass": 1}
    none = [ScaleConnectionSpec(n=0), GroundSpringSpec(n=(0, 0)), ScaleMassSpec(n=0)]
    assert counts(sample_system(seed, "chain", 8, feature_types=none)) == {}


def test_int_and_equal_range_give_identical_systems():
    a = sample_system(3, "tower", 6, feature_types=[ScaleConnectionSpec(n=2)])
    b = sample_system(3, "tower", 6, feature_types=[ScaleConnectionSpec(n=(2, 2))])
    assert a.features == b.features


@pytest.mark.parametrize("bad", [(2, 1), -1, "a", (1, 2, 3), 1.5, (-1, 2)])
def test_invalid_counts_raise(bad):
    with pytest.raises(ValueError):
        sample_system(0, "tower", 5, feature_types=[ScaleMassSpec(n=bad)])


def test_counts_are_capped_and_targets_distinct():
    rs = sample_system(0, "tower", 3, feature_types=[ScaleMassSpec(n=10)])
    assert sorted(f.dof for f in rs.features) == [0, 1, 2]


def test_spec_ranges_are_respected():
    r = sample_system(
        3,
        "chain",
        8,
        feature_types=[
            ScaleConnectionSpec(n=2, k_range=(0.25, 0.5)),
            ScaleMassSpec(n=3, factor_range=(2, 3)),
        ],
    )
    sc = [f for f in r.features if isinstance(f, ScaleConnection)]
    ms = [f for f in r.features if isinstance(f, ScaleMass)]
    assert len(sc) == 2 and len(ms) == 3
    assert all(0.25 <= f.k_factor <= 0.5 for f in sc) and all(2 <= f.factor <= 3 for f in ms)
    assert len({f.index for f in sc}) == 2 and len({f.dof for f in ms}) == 3


def test_ground_spring_damping_option():
    plain = sample_system(0, "tower", 5, feature_types=[GroundSpringSpec(n=2)])
    coupled = sample_system(0, "tower", 5, feature_types=[GroundSpringSpec(n=2, couple_damping=True)])
    assert len(plain.features) == len(coupled.features) == 2
    assert all(f.c == 0 for f in plain.features)  # spring only by default
    assert all(np.isclose(f.c / f.k, 2 / 2500) for f in coupled.features)
    assert [f.k for f in plain.features] == [f.k for f in coupled.features]


# ---- appendage sampling ---------------------------------------------------------------
@pytest.mark.parametrize("seed", range(20))
def test_default_appendage_is_one_on_the_top_floor(seed):
    rs = sample_system(seed, "tower", 6, feature_types=["appendage"])
    (f,) = rs.features
    assert isinstance(f, AddAppendage) and f.attach == 5
    assert 0.5 <= f.m <= 5.0 and 250 <= f.k <= 5000 and np.isclose(f.c / f.k, 2 / 2500)
    assert rs.system.ndof == 7


def test_appendage_attach_options():
    r = sample_system(1, "chain", 6, feature_types=[AppendageSpec(n=2, attach="top")])
    assert len(r.features) == 1  # only one candidate floor
    r = sample_system(1, "chain", 6, feature_types=[AppendageSpec(n=3, attach="random")])
    assert len({f.attach for f in r.features}) == 3 and r.system.ndof == 9
    assert sample_system(1, "tower", 6, feature_types=[AppendageSpec(attach=2)]).features[0].attach == 2
    r = sample_system(1, "tower", 6, feature_types=[AppendageSpec(attach=[1, 4], n=2)])
    assert {f.attach for f in r.features} == {1, 4}
    r = sample_system(1, "tower", 6, feature_types=[AppendageSpec(couple_damping=False, m_range=(2, 2))])
    assert r.features[0].c == 0 and np.isclose(r.features[0].m, 10)


@pytest.mark.parametrize(
    "bad", [AppendageSpec(attach=9), AppendageSpec(attach="bottom"), AppendageSpec(attach=[0, 7])]
)
def test_invalid_appendage_attach_raises(bad):
    with pytest.raises((IndexError, ValueError)):
        sample_system(0, "tower", 6, feature_types=[bad])


# ---- general properties --------------------------------------------------------------
ALL_TYPES = ["scale", "ground", "mass", AppendageSpec(attach="random", n=(0, 2))]


@pytest.mark.parametrize("kind", ["tower", "chain"])
@pytest.mark.parametrize("seed", range(75))
def test_sampled_systems_are_physical(kind, seed):
    s = sample_system(seed, kind, 6, feature_types=ALL_TYPES).system
    np.testing.assert_allclose(s.K, s.K.T)
    np.testing.assert_allclose(s.C, s.C.T)
    assert np.linalg.eigvalsh(s.K).min() > 0 and np.diag(s.M).min() > 0
    A, _ = make_state_matrices(s)
    assert np.linalg.eigvals(A).real.max() < 0  # stable


def test_same_seed_same_system():
    a = sample_system(7, "chain", 5, feature_types=ALL_TYPES)
    b = sample_system(7, "chain", 5, feature_types=ALL_TYPES)
    assert a.features == b.features
    np.testing.assert_array_equal(a.system.K, b.system.K)
    np.testing.assert_array_equal(a.system.M, b.system.M)


def test_generator_can_be_passed_instead_of_seed():
    rng = np.random.default_rng(5)
    first, second = sample_system(rng, "tower", 5), sample_system(rng, "tower", 5)
    assert first.features != second.features  # the generator advances


# ---- targets: choose where a change happens -------------------------------------------
@pytest.mark.parametrize("seed", range(20))
def test_single_target_always_changes_exactly_that_place(seed):
    rs = sample_system(
        seed,
        "tower",
        6,
        feature_types=[
            ScaleConnectionSpec(targets=3, k_range=(0.3, 0.6)),
            ScaleMassSpec(targets=2),
            GroundSpringSpec(targets=5),
        ],
    )
    assert counts(rs) == {"ScaleConnection": 1, "ScaleMass": 1, "AddGroundSpring": 1}
    sc, ms, gs = rs.features
    assert sc.index == 3 and 0.3 <= sc.k_factor <= 0.6
    assert ms.dof == 2 and gs.dof == 5


def test_targets_without_n_changes_every_target():
    rs = sample_system(0, "tower", 6, feature_types=[ScaleMassSpec(targets=[1, 4, 5], factor_range=(2, 2))])
    assert sorted(f.dof for f in rs.features) == [1, 4, 5]
    assert all(f.factor == 2.0 for f in rs.features)
    np.testing.assert_allclose(np.diag(rs.system.M), [5, 10, 5, 5, 10, 10])


@pytest.mark.parametrize("seed", range(20))
def test_targets_with_n_picks_a_distinct_subset_of_the_targets(seed):
    rs = sample_system(seed, "chain", 6, feature_types=[ScaleConnectionSpec(targets=[0, 2, 4, 6], n=2)])
    idx = [f.index for f in rs.features]
    assert len(idx) == 2 and len(set(idx)) == 2 and set(idx) <= {0, 2, 4, 6}
    rs = sample_system(seed, "chain", 6, feature_types=[GroundSpringSpec(targets=[1, 3], n=(0, 1))])
    assert len(rs.features) <= 1 and all(f.dof in (1, 3) for f in rs.features)


def test_targeted_change_alters_only_the_targeted_connection():
    base = make_mdof_tower(5)
    rs = sample_system(1, "tower", 5, feature_types=[ScaleConnectionSpec(targets=2, k_range=(2, 2))])
    expected = np.zeros((5, 5))
    expected[1:3, 1:3] = 5000 * np.array([[1, -1], [-1, 1]])  # connection 2 joins floors 1 and 2
    np.testing.assert_allclose(rs.system.K - base.K, expected)


def test_n_larger_than_targets_is_capped_and_duplicates_ignored():
    rs = sample_system(0, "tower", 6, feature_types=[ScaleMassSpec(targets=[2, 2, 3], n=10)])
    assert sorted(f.dof for f in rs.features) == [2, 3]


def test_target_forms_accepted():
    for targets in (2, np.int64(2), [2], (2,), np.array([2])):
        rs = sample_system(0, "tower", 5, feature_types=[ScaleMassSpec(targets=targets)])
        assert [f.dof for f in rs.features] == [2]


def test_connection_targets_depend_on_kind():
    # towers have connections 0..ndof-1, chains one more (the top-to-ground link)
    with pytest.raises(IndexError, match="connection targets"):
        sample_system(0, "tower", 4, feature_types=[ScaleConnectionSpec(targets=4)])
    rs = sample_system(0, "chain", 4, feature_types=[ScaleConnectionSpec(targets=4)])
    assert rs.features[0].index == 4


@pytest.mark.parametrize(
    "bad, exc",
    [
        ("top", TypeError),
        (1.5, TypeError),
        ([1, 2.0], TypeError),
        (True, TypeError),
        (5.0, TypeError),
        ([], ValueError),
        ([7], IndexError),
        ([-1], IndexError),
        ([0, 6], IndexError),
    ],
)
def test_invalid_targets_raise(bad, exc):
    with pytest.raises(exc):
        sample_system(0, "tower", 6, feature_types=[ScaleMassSpec(targets=bad)])


def test_default_n_without_targets_is_unchanged():
    # no targets and no n: the old (0, 2) behaviour, i.e. between 0 and 2 changes
    sizes = {len(sample_system(s, "tower", 6, feature_types=[ScaleMassSpec()]).features) for s in range(60)}
    assert sizes == {0, 1, 2}


# ---- stiffness and damping are controlled separately ---------------------------------------
def scaled(spec, seed=0, kind="tower", ndof=8):
    rs = sample_system(seed, kind, ndof, feature_types=[spec])
    return [f for f in rs.features if isinstance(f, ScaleConnection)]


@pytest.mark.parametrize("seed", range(20))
def test_default_sampling_never_changes_damping(seed):
    # The default feature types leave C exactly as in the baseline, for both kinds of system
    for kind, ref in (("tower", make_mdof_tower), ("chain", make_mdof_chain)):
        rs = sample_system(seed, kind, 6)
        np.testing.assert_array_equal(rs.system.C, ref(6).C)


def test_scale_connection_spec_modes():
    sc = scaled(ScaleConnectionSpec(n=4, k_range=(0.5, 2)))  # stiffness only
    assert len(sc) == 4 and all(f.c_factor == 1.0 and f.k_factor != 1.0 for f in sc)

    sc = scaled(ScaleConnectionSpec(n=4))  # neither given: stiffness only, default range
    assert len(sc) == 4 and all(f.c_factor == 1.0 and 0.5 <= f.k_factor <= 2.0 for f in sc)

    sc = scaled(ScaleConnectionSpec(n=4, c_range=(0.5, 2)))  # damping only
    assert len(sc) == 4 and all(f.k_factor == 1.0 and f.c_factor != 1.0 for f in sc)

    sc = scaled(ScaleConnectionSpec(n=4, k_range=(2, 3), c_range=(0.2, 0.3)))  # independent
    assert all(2 <= f.k_factor <= 3 and 0.2 <= f.c_factor <= 0.3 for f in sc)

    sc = scaled(ScaleConnectionSpec(n=4, k_range=(0.5, 2), couple_damping=True))  # opt-in coupling
    assert all(f.c_factor == f.k_factor != 1.0 for f in sc)
    sc = scaled(ScaleConnectionSpec(n=4, couple_damping=True))  # coupling with the default range
    assert all(f.c_factor == f.k_factor != 1.0 for f in sc)


def test_damping_only_change_alters_only_C():
    base = make_mdof_tower(5)
    rs = sample_system(1, "tower", 5, feature_types=[ScaleConnectionSpec(targets=2, c_range=(2, 2))])
    np.testing.assert_array_equal(rs.system.K, base.K)
    expected = np.zeros((5, 5))
    expected[1:3, 1:3] = 4 * np.array([[1, -1], [-1, 1]])  # connection 2: extra c = 2 x baseline 2c
    np.testing.assert_allclose(rs.system.C - base.C, expected)


def test_contradictory_or_invalid_scale_connection_options_raise():
    with pytest.raises(ValueError, match="couple_damping"):
        ScaleConnectionSpec(c_range=(0.5, 2), couple_damping=True)
    for bad in ((2, 1), (0, 2), (-1, 2), (1, 2, 3), 2, "ab", (float("nan"), 2)):
        with pytest.raises(ValueError, match="k_range"):
            ScaleConnectionSpec(k_range=bad)
        with pytest.raises(ValueError, match="c_range"):
            ScaleConnectionSpec(c_range=bad)
    with pytest.raises(TypeError):  # the old name no longer exists
        ScaleConnectionSpec(factor_range=(0.5, 2))


def test_ranges_are_validated_on_every_spec():
    for make in (
        lambda r: GroundSpringSpec(k_range=r),
        lambda r: GroundDamperSpec(c_range=r),
        lambda r: ScaleMassSpec(factor_range=r),
        lambda r: AppendageSpec(m_range=r),
        lambda r: AppendageSpec(k_range=r),
    ):
        for bad in ((2, 1), (0, 1), "ab", 3):
            with pytest.raises(ValueError, match="range"):
                make(bad)
        make((1, 1))  # equal ends are allowed and fix the magnitude


# ---- ground dampers --------------------------------------------------------------------------
def test_ground_damper_matrices_and_description():
    base = make_mdof_tower(4)
    rs = build_system("tower", 4, features=[AddGroundDamper(2, 1.5)])
    np.testing.assert_array_equal(rs.system.K, base.K)  # no stiffness
    expected = np.zeros((4, 4))
    expected[2, 2] = 1.5
    np.testing.assert_allclose(rs.system.C - base.C, expected)
    assert rs.describe() == ["ground damper at floor 2: c=1.5"]
    with pytest.raises(IndexError):
        build_system("tower", 4, features=[AddGroundDamper(4, 1.0)])


@pytest.mark.parametrize("kind, c_conn", [("tower", 4.0), ("chain", 2.0)])
def test_ground_damper_spec_scales_with_baseline_connection_damping(kind, c_conn):
    rs = sample_system(3, kind, 6, feature_types=[GroundDamperSpec(n=3, c_range=(0.5, 0.5))])
    assert counts(rs) == {"AddGroundDamper": 3}
    assert all(np.isclose(f.c, 0.5 * c_conn) for f in rs.features)
    assert len({f.dof for f in rs.features}) == 3


def test_ground_damper_targets_and_name():
    rs = sample_system(0, "tower", 6, feature_types=[GroundDamperSpec(targets=[1, 4])])
    assert sorted(f.dof for f in rs.features) == [1, 4]
    rs = sample_system(0, "tower", 6, feature_types=["ground_damper"])
    assert all(isinstance(f, AddGroundDamper) for f in rs.features)
    # not part of the defaults
    defaults = [f for s in range(40) for f in sample_system(s, "tower", 6).features]
    assert not any(isinstance(f, AddGroundDamper) for f in defaults)


def test_ground_spring_and_damper_on_one_floor_are_in_parallel():
    base = make_mdof_chain(4)
    rs = build_system("chain", 4, features=[AddGroundSpring(1, 500), AddGroundDamper(1, 0.7)])
    assert np.isclose(rs.system.K[1, 1] - base.K[1, 1], 500)
    assert np.isclose(rs.system.C[1, 1] - base.C[1, 1], 0.7)


def test_describe_only_lists_what_changed():
    rs = build_system(
        "tower", 4, features=[ScaleConnection(1, 2.0), ScaleConnection(2, 1.0, 0.5), ScaleConnection(3)]
    )
    assert rs.describe() == [
        "connection 1: k x2",
        "connection 2: c x0.5",
        "connection 3: unchanged",
    ]
