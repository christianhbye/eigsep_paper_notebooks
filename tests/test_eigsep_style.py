import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import pytest

import eigsep_style as es


def box_pt(path):
    """(width, height) of a PDF's page box in PostScript points, via pdfinfo."""
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", out)
    assert m, f"pdfinfo gave no page size for {path}:\n{out}"
    return float(m.group(1)), float(m.group(2))


def test_column_width_is_244_tex_points():
    assert es.COL_IN == pytest.approx(244.0 / 72.27, abs=1e-9)


def test_text_width_is_508_tex_points():
    assert es.FULL_IN == pytest.approx(508.0 / 72.27, abs=1e-9)


def test_figsize_one_column_uses_column_width():
    w, h = es.figsize(cols=1, aspect=0.5)
    assert w == pytest.approx(es.COL_IN)
    assert h == pytest.approx(es.COL_IN * 0.5)


def test_figsize_two_column_uses_text_width():
    w, _ = es.figsize(cols=2)
    assert w == pytest.approx(es.FULL_IN)


def test_height_in_overrides_aspect():
    w, h = es.figsize(cols=1, aspect=0.75, height_in=3.9720)
    assert (w, h) == pytest.approx((es.COL_IN, 3.9720))


def test_saved_box_equals_figsize(tmp_path):
    """The contract: the PDF box is identically the figure size."""
    fig, _ = es.figure(cols=1, height_in=3.9720)
    out = tmp_path / "g.pdf"
    es.save(fig, out)
    assert box_pt(out) == pytest.approx((es.COL_IN * 72, 3.9720 * 72), abs=0.1)


def test_saved_box_equals_figsize_two_column(tmp_path):
    fig, _ = es.figure(cols=2, height_in=2.6833)
    out = tmp_path / "g2.pdf"
    es.save(fig, out)
    assert box_pt(out) == pytest.approx((es.FULL_IN * 72, 2.6833 * 72), abs=0.1)


def test_save_refuses_bbox_inches(tmp_path):
    """The signature is the enforcement point -- tight bbox is unreachable."""
    fig, _ = es.figure(cols=1)
    with pytest.raises(TypeError):
        es.save(fig, tmp_path / "x.pdf", bbox_inches="tight")


def test_apply_rc_sets_the_paper_font_scheme():
    es.apply_rc()
    rc = matplotlib.rcParams
    assert rc["axes.labelsize"] == 8
    assert rc["xtick.labelsize"] == 7
    assert rc["ytick.labelsize"] == 7
    assert rc["legend.fontsize"] == 6.5
    assert rc["savefig.dpi"] == 600
    assert rc["savefig.bbox"] is None
