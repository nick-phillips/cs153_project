"""Tests for the shared matplotlib style helpers."""

from pathlib import Path

from biomarker_agent import plotstyle


def test_palette_and_cycle():
    assert plotstyle.PALETTE["positive"] != plotstyle.PALETTE["negative"]
    assert len(plotstyle.OKABE_ITO) >= 7
    # all palette/cycle colors are hex strings
    for c in plotstyle.OKABE_ITO:
        assert c.startswith("#") and len(c) == 7


def test_apply_style_sets_publication_rcparams():
    plotstyle.apply_style()
    import matplotlib as mpl
    assert mpl.get_backend().lower() == "agg"
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["savefig.dpi"] == 110


def test_finalize_writes_png_and_closes(tmp_path):
    plotstyle.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "f.png"
    returned = plotstyle.finalize(fig, out)
    assert Path(returned) == out
    assert out.exists() and out.stat().st_size > 500
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature
    assert plt.fignum_exists(fig.number) is False  # figure closed
