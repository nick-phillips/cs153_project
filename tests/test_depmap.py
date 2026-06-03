"""Tests for the local DepMap dependency tool."""

from biomarker_agent.datactx import DataContext
from biomarker_agent.tools import depmap


def test_depmap_tool(synthetic_data):
    ff, rf, cid = synthetic_data
    tool = depmap.make_tool(DataContext(ff, rf))
    out = tool.run({"gene": "BBB"})
    assert out["profile"]["is_selective"] is True
    assert "codependencies" in out
