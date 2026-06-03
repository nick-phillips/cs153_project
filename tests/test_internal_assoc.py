"""Tests for the internal-association tool wrapper."""

from biomarker_agent.datactx import DataContext
from biomarker_agent.tools import internal_assoc


def test_tool_wraps_datactx(synthetic_data):
    ff, rf, cid = synthetic_data
    tool = internal_assoc.make_tool(DataContext(ff, rf))
    assert tool.name == "internal_association"
    out = tool.run({"feature_name": "GE_AAA", "compound_id": cid})
    assert out["pearson_r"] > 0.7
