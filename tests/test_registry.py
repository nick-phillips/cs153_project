"""Tests for assembling the tool registry."""


from biomarker_agent.datactx import DataContext
from biomarker_agent.tools import build_registry


def test_build_registry(synthetic_data, tmp_path):
    ff, rf, cid = synthetic_data
    ti = tmp_path / "ti.csv"
    ti.write_text("IDs,Drug.Name,MOA,repurposing_target\nBRD:TEST-1,DRUG,MOA,GENE\n")
    reg = build_registry(
        data_ctx=DataContext(ff, rf),
        treatment_info=ti,
        cache_dir=tmp_path / "cache",
        literature_backend="pubmed",
    )
    names = set(reg.names())
    assert {
        "drug_context", "internal_association", "depmap_dependency", "string_enrichment",
        "opentargets_target", "cbioportal_mutations", "reactome_pathways", "literature_search",
    } <= names
    schemas = reg.anthropic_schemas()
    assert all("input_schema" in s for s in schemas)
    # dispatch works and degrades gracefully
    out = reg.dispatch("drug_context", {"compound_id": "BRD:TEST-1"})
    assert out["drug_name"] == "DRUG"


def test_build_registry_with_figures(synthetic_data, tmp_path):
    from biomarker_agent.loader import CompoundResult
    ff, rf, cid = synthetic_data
    ti = tmp_path / "ti.csv"
    ti.write_text("IDs,Drug.Name,MOA,repurposing_target\nBRD:TEST-1,DRUG,MOA,GENE\n")
    cr = CompoundResult(compound_id=cid, dir_name="d", path=None, n_samples=60,
                        metrics={}, passing_features=[], passing_by_class={})
    reg = build_registry(
        data_ctx=DataContext(ff, rf), treatment_info=ti, cache_dir=tmp_path / "c",
        figures_dir=tmp_path / "out" / "figures", compound_result=cr,
    )
    names = set(reg.names())
    assert "plot_feature_response" in names
    assert "plot_string_network" in names
    # data tools still present
    assert "internal_association" in names
