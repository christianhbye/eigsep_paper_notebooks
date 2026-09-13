"""Every deposited figure's PDF box, against the target table in the spec.

A figure that has not been retrofitted yet fails here. That is intended: the
suite is the running score for the retrofit.
"""
import re
import subprocess
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parent.parent

# figsize x 72, from docs/figure-workflow-spec.md section 7.
TARGETS = {
    "horizon_perturbations_1col.pdf": (243.09, 285.98),
    "beam_comparison.pdf":            (506.10, 193.20),
    "horizon_shift.pdf":              (506.10, 253.67),
    "beam_dBi.pdf":                   (243.09, 395.15),
    "reflections.pdf":                (243.09, 172.32),
    "s11_sim_meas.pdf":               (243.09, 317.97),
    "rfi_waterfall.pdf":              (243.09, 311.09),
    "point_src.pdf":                  (243.09, 311.09),
    "beam_modulation.pdf":            (243.09, 285.98),
}


def box_pt(path):
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", out)
    assert m, f"pdfinfo gave no page size for {path}:\n{out}"
    return float(m.group(1)), float(m.group(2))


@pytest.mark.parametrize("name,expected", sorted(TARGETS.items()))
def test_figure_box_matches_target(name, expected):
    path = NB / name
    assert path.exists(), f"{name} has not been deposited by its notebook yet"
    assert box_pt(path) == pytest.approx(expected, abs=0.1)
