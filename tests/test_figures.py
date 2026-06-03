"""Tests for the figure Tool wrappers."""

from pathlib import Path

from biomarker_agent.cache import DiskCache
from biomarker_agent.datactx import DataContext
from biomarker_agent.loader import CompoundResult, PassingFeature
from biomarker_agent.tools import figures

PNG = b"\x89PNG\r\n\x1a\n"


def _png_ok(out_dir, rel):
    p = Path(out_dir) / rel
    return p.exists() and p.read_bytes()[:8] == PNG


def _compound_result():
    return CompoundResult(
        compound_id="BRD:TEST-1", dir_name="BRD_TEST-1", path=None, n_samples=60,
        metrics={}, passing_features=[
            PassingFeature("CRISPR_BBB", "CRISPR", "BBB", 0.7, 0.02, 0.002, 1e-9, 1e-6),
            PassingFeature("GE_AAA", "GE", "AAA", 0.6, 0.01, 0.003, 1e-6, 1e-4),
        ], passing_by_class={"CRISPR": 1, "GE": 1})


def _tools(tmp_path, synthetic_data):
    ff, rf, cid = synthetic_data
    figs_dir = tmp_path / "out" / "figures"
    reg = figures.make_figure_tools(
        figures_dir=figs_dir, rel_prefix="figures",
        data_ctx=DataContext(ff, rf), compound_result=_compound_result(),
        cache=DiskCache(tmp_path / "cache"),
    )
    return {t.name: t for t in reg}, tmp_path / "out"


def test_make_figure_tools_names(tmp_path, synthetic_data):
    tools, _ = _tools(tmp_path, synthetic_data)
    assert {
        "plot_feature_response", "plot_feature_panel", "plot_dependency_distribution",
        "plot_codependency_bar", "plot_passing_importance", "plot_string_network",
        "plot_mutation_frequency", "plot_pathway_membership",
    } == set(tools)


def test_feature_response_tool(tmp_path, synthetic_data):
    tools, out = _tools(tmp_path, synthetic_data)
    res = tools["plot_feature_response"].run({"feature": "GE_AAA"})
    assert res["figure"].startswith("figures/")
    assert "caption" in res
    assert _png_ok(out, res["figure"])


def test_dependency_tool(tmp_path, synthetic_data):
    tools, out = _tools(tmp_path, synthetic_data)
    res = tools["plot_dependency_distribution"].run({"gene": "BBB"})
    assert _png_ok(out, res["figure"])


def test_passing_importance_tool(tmp_path, synthetic_data):
    tools, out = _tools(tmp_path, synthetic_data)
    res = tools["plot_passing_importance"].run({})
    assert _png_ok(out, res["figure"])


def test_feature_response_unknown_feature_errors(tmp_path, synthetic_data):
    tools, _ = _tools(tmp_path, synthetic_data)
    res = tools["plot_feature_response"].run({"feature": "GE_NOPE"})
    assert "error" in res


def test_mutation_frequency_tool_mocked(tmp_path, synthetic_data, monkeypatch):
    from biomarker_agent.tools import cbioportal
    monkeypatch.setattr(cbioportal, "make_tool",
                        lambda cache: _FakeTool({"gene": "X", "n_mutated_samples": 5}))
    tools, out = _tools(tmp_path, synthetic_data)
    res = tools["plot_mutation_frequency"].run({"genes": ["AAA", "BBB"]})
    assert _png_ok(out, res["figure"])


class _FakeTool:
    def __init__(self, payload):
        self._p = payload

    def run(self, args):
        return dict(self._p, gene=args.get("gene", "X"))
