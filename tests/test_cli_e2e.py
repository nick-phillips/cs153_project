"""End-to-end CLI test using a fake client injected via run_one."""

from pathlib import Path

from biomarker_agent import cli


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Msg:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason


class FakeClient:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        return _Msg([_Block(type="tool_use", id="t", name="submit_report", input={
            "summary": "ok",
            "hypotheses": [{"rank": 1, "title": "H", "features": ["CRISPR_SMARCD1"],
                            "mechanism": "m", "novelty": "off-MOA", "confidence": 0.5}],
        })])


def test_run_one_writes_report(sample_compound, tmp_path):
    data = Path("data")
    out = cli.run_one(
        compound_dir=sample_compound,
        out_dir=tmp_path / "out",
        feature_file=data / "x-all_v4.pkl",
        response_file=data / "responses_primary_v4.pkl",
        treatment_info=data / "primary_screen_treatment_info.csv",
        cache_dir=tmp_path / "cache",
        client=FakeClient(),
        model="fake",
        literature_backend="pubmed",
        max_tool_calls=5,
    )
    assert out["markdown"].exists()
    assert "BRD:BRD-K25244359-066-03-4" in out["markdown"].read_text()
    # trace is persisted alongside the report
    import json
    assert out["trace"].exists()
    trace = json.loads(out["trace"].read_text())
    assert trace["compound_id"] == "BRD:BRD-K25244359-066-03-4"
    assert "transcript" in trace
