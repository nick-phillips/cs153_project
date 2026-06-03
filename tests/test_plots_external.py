"""Tests for plotting functions that visualize external-tool outputs."""

from pathlib import Path

from biomarker_agent import plots

PNG = b"\x89PNG\r\n\x1a\n"


def _is_png(p):
    p = Path(p)
    return p.exists() and p.stat().st_size > 500 and p.read_bytes()[:8] == PNG


def test_string_network(tmp_path):
    interactions = [{"a": "TP53", "b": "MDM2", "score": 0.99},
                    {"a": "TP53", "b": "CDKN1A", "score": 0.9}]
    out = plots.string_network(["TP53", "MDM2", "CDKN1A"], interactions, tmp_path / "sn.png")
    assert _is_png(out)


def test_string_network_no_edges_still_plots_nodes(tmp_path):
    out = plots.string_network(["A", "B"], [], tmp_path / "sn2.png")
    assert _is_png(out)


def test_mutation_frequency(tmp_path):
    out = plots.mutation_frequency([("TP53", 4538), ("ITGA1", 0)], tmp_path / "mf.png")
    assert _is_png(out)


def test_pathway_membership(tmp_path):
    mapping = {"ITGA1": ["Integrin interactions", "Laminin interactions"],
               "PDLIM5": ["Integrin interactions"]}
    out = plots.pathway_membership(mapping, tmp_path / "pm.png")
    assert _is_png(out)


def test_pathway_membership_empty_errors(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        plots.pathway_membership({}, tmp_path / "x.png")
