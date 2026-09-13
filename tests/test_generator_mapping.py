"""Which generator produces which figure from which data.

Read from the code, never inferred from filenames. s11-plot.ipynb also loads
s11.npz and sparams.csv, but only for exploratory plots that run with
save=False and write a bowtie_s11.pdf that never reaches the paper -- which is
what makes the wrong inference look right.
"""
import json
import re
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parent.parent

MAPPING = {
    "beam_comparison.ipynb":      ("beam_comparison.pdf",            ("beam_comparison.npz", "21cm_models.npz")),
    "s11-plot.ipynb":             ("reflections.pdf",                ("s11_terrain.npz",)),
    "s11_sim_meas.ipynb":         ("s11_sim_meas.pdf",               ("s11.npz", "sparams.csv")),
    "beam.ipynb":                 ("beam_dBi.pdf",                   ("beam_maps.npz",)),
    "beam_modulation.ipynb":      ("beam_modulation.pdf",            ("beam_modulation.npz",)),
    "horizon_perturbations.ipynb":("horizon_perturbations_1col.pdf", ("horizon_perturbations.npz",)),
    "horizon_shift.ipynb":        ("horizon_shift.pdf",              ("horizon_shift.npz", "foreground_svd.npz")),
    "rfi_waterfall.ipynb":        ("rfi_waterfall.pdf",              ("rfi_waterfall.npz",)),
    "virA_2dfilter.ipynb":        ("point_src.pdf",                  ("virA_fringes.npz",)),
}


def source_of(name):
    path = NB / name
    if not path.exists():
        pytest.skip(f"{name} does not exist yet")
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        return "\n".join("".join(c["source"]) for c in nb["cells"]
                          if c["cell_type"] == "code")
    return path.read_text()


@pytest.mark.parametrize("gen,expected", sorted(MAPPING.items()))
def test_generator_writes_the_expected_figure(gen, expected):
    pdf, _ = expected
    src = source_of(gen)
    active = [l for l in src.split("\n") if not l.strip().startswith("#")]
    assert any(pdf in l for l in active), f"{gen} does not write {pdf}"


@pytest.mark.parametrize("gen,expected", sorted(MAPPING.items()))
def test_generator_loads_the_expected_data(gen, expected):
    _, inputs = expected
    src = source_of(gen)
    for npz in inputs:
        assert npz in src, f"{gen} does not load {npz}"


@pytest.mark.parametrize("gen,expected", sorted(MAPPING.items()))
def test_generator_never_saves_with_a_tight_bbox(gen, expected):
    src = source_of(gen)
    active = [l for l in src.split("\n") if not l.strip().startswith("#")]
    offenders = [l.strip() for l in active
                 if "savefig" in l and ("bbox_inches" in l or "pad_inches" in l)]
    assert not offenders, f"{gen} still saves with a tight bbox: {offenders}"


def test_no_generator_reads_an_absolute_path():
    for gen in MAPPING:
        src = source_of(gen)
        assert "/home/" not in src, f"{gen} reads an absolute path; it is not standalone"


def test_no_generator_ships_an_absolute_path_in_its_stored_output():
    """Source cells were checked above; the deposit also ships the outputs.

    virA_2dfilter carried two site-packages warnings naming a local venv, which
    the source-only check could never see.
    """
    for gen in MAPPING:
        path = NB / gen
        if not path.exists() or path.suffix != ".ipynb":
            continue
        nb = json.loads(path.read_text())
        for cell in nb["cells"]:
            for out in cell.get("outputs", []):
                blob = json.dumps(out)
                assert "/home/" not in blob, \
                    f"{gen} stores an absolute path in its output"


@pytest.mark.parametrize("gen", sorted(MAPPING))
def test_generator_seeds_every_random_draw(gen):
    """An unseeded draw republishes a different figure on every run.

    virA_2dfilter inpaints flagged pixels with a noise realisation, and the
    delay transform spreads each flagged channel over every delay -- so an
    unseeded draw moved the whole of point_src.pdf, not just the gaps.
    """
    src = source_of(gen)
    active = [l for l in src.split("\n") if not l.strip().startswith("#")]

    legacy = [l.strip() for l in active
              if re.search(r"np\.random\.(?!default_rng\b)", l)]
    assert not legacy, f"{gen} draws from the unseeded global RNG: {legacy}"

    unseeded = [l.strip() for l in active
                if re.search(r"default_rng\(\s*\)", l)]
    assert not unseeded, f"{gen} calls default_rng() with no seed: {unseeded}"
