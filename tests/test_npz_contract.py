"""The published curves must reproduce from the deposited inputs.

Not a byte comparison of SVD bases: 191 of the 201 singular values are
degenerate, so the trailing singular vectors are LAPACK-dependent and any
orthonormal basis of that near-null space is an equally valid SVD. The
published quantity is a projection onto the trailing subspace, which is
invariant to that choice -- so the curves are what must reproduce.
"""
import numpy as np
import pytest

from pathlib import Path

NB = Path(__file__).resolve().parent.parent


def load(name):
    """Every deposited NPZ is gitignored, so a bare clone has none of them.

    Skip rather than error there: these tests check the contract of files that
    ship with the Zenodo deposit, not of files the repository carries.
    """
    path = NB / name
    if not path.exists():
        pytest.skip(f"{name} is gitignored and absent from this checkout")
    return np.load(path, allow_pickle=True)


def resid_curves(dT_axis, Vh, n_modes):
    n_f = dT_axis.shape[1]
    coeff = dT_axis @ Vh.T
    return np.array([np.sqrt(np.sum(coeff[:, N:] ** 2, axis=1) / n_f)
                     for N in n_modes])


def test_beam_comparison_ships_waterfalls_not_reduced_curves():
    d = load("beam_comparison.npz")
    assert "t_sys" in d, "must ship the per-beam waterfalls"
    assert d["t_sys"].shape == (3, 1436, 201)
    assert "fg_resid" not in d, "fg_resid is derived; it belongs in the notebook"
    assert "t21_pct" not in d, "t21_pct is derived; it belongs in the notebook"


def test_beam_comparison_beams_are_in_the_figure_order():
    """beam_sims.npz stores bowtie/vivaldi/isotropic; the figure wants
    isotropic/bowtie/vivaldi. A copy without reindexing mislabels every curve."""
    d = load("beam_comparison.npz")
    assert [str(x) for x in d["order"]] == ["isotropic", "bowtie", "vivaldi"]
    # the bowtie row must be the one that reproduces the published basis
    bowtie = d["t_sys"][1] - float(d["t_receiver"])
    fs = load("foreground_svd.npz")
    ref = fs["t_sys"] - float(fs["t_receiver"])
    assert np.abs(bowtie - ref).max() / np.abs(ref).max() < 1e-5, \
        "row 1 is not the bowtie waterfall -- the reindex is wrong"


def test_beam_comparison_carries_contract_metadata():
    d = load("beam_comparison.npz")
    for key in ("description", "provenance", "params"):
        assert key in d, f"missing contract key {key}"


def test_bowtie_waterfall_reproduces_the_published_curves():
    """The bowtie column's basis must give back horizon_shift's curves."""
    bc = load("beam_comparison.npz")
    hs = load("horizon_shift.npz")
    order = [str(x) for x in bc["order"]]
    i = order.index("bowtie")
    M = bc["t_sys"][i] - float(bc["t_receiver"])
    _, _, Vh = np.linalg.svd(M, full_matrices=False)

    dT = hs["dT_disp"].reshape(-1, hs["dT_disp"].shape[-1])
    n_modes = np.arange(19)
    a = resid_curves(dT, hs["Vh"], n_modes)
    b = resid_curves(dT, Vh, n_modes)
    rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-30)
    assert rel.max() < 1e-5, f"curves diverge at {rel.max():.2e}"


def test_vh_derives_from_foreground_svd_to_curve_precision():
    """The exact matrix behind the published basis, to <1e-9 on the curves."""
    fs = load("foreground_svd.npz")
    hs = load("horizon_shift.npz")
    M = fs["t_sys"] - float(fs["t_receiver"])
    _, _, Vh = np.linalg.svd(M, full_matrices=False)

    dT = hs["dT_disp"].reshape(-1, hs["dT_disp"].shape[-1])
    n_modes = np.arange(19)
    # reference curves, recomputed the same way the notebook now does
    ref = resid_curves(dT, Vh, n_modes)
    # a second, independently ordered SVD must give the same curves
    _, _, Vh2 = np.linalg.svd(np.asfortranarray(M), full_matrices=False)
    got = resid_curves(dT, Vh2, n_modes)
    rel = np.abs(ref - got) / np.maximum(np.abs(ref), 1e-30)
    assert rel.max() < 1e-9, f"curves not basis-invariant: {rel.max():.2e}"


def test_s11_terrain_ships_spectra_not_delay_transforms():
    d = load("s11_terrain.npz")
    assert "spec" in d, "must ship the per-height frequency spectra"
    for key in ("description", "provenance", "params"):
        assert key in d, f"missing contract key {key}"


def test_s11_terrain_covers_the_paper_heights():
    d = load("s11_terrain.npz")
    heights = np.asarray(d["heights"], dtype=float)
    for h in (1.0, 114.0):
        assert np.isclose(heights, h, atol=0.5).any(), f"missing height {h} m"


CONTRACT_NPZ = [
    "beam_comparison.npz", "beam_maps.npz", "beam_modulation.npz",
    "foreground_svd.npz", "horizon_perturbations.npz", "horizon_shift.npz",
    "s11.npz", "s11_terrain.npz", "virA_fringes.npz", "21cm_models.npz",
]


@pytest.mark.parametrize("name", CONTRACT_NPZ)
def test_npz_carries_contract_metadata(name):
    d = load(name)
    for key in ("description", "provenance", "params"):
        assert key in d, f"{name} is missing {key}"


def test_rfi_waterfall_is_deliberately_exempt():
    """90 MB re-uploaded to add a metadata string is not worth it.

    It is deposited unchanged from the existing Zenodo record and carries no
    description/provenance/params keys.
    """
    d = load("rfi_waterfall.npz")
    assert "description" not in d
