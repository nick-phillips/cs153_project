"""Analysis tools available to the interpretation agent + registry assembly."""

from dataclasses import dataclass
from pathlib import Path

from ..cache import DiskCache
from ..datactx import DataContext
from .base import Tool
from . import (
    cbioportal,
    depmap,
    drug_context,
    internal_assoc,
    literature,
    opentargets,
    pathways,
    stringdb,
)
from . import figures as figures_mod


@dataclass
class Registry:
    tools: dict[str, Tool]  # name -> Tool

    def names(self) -> list[str]:
        return list(self.tools)

    def anthropic_schemas(self) -> list:
        return [t.to_anthropic() for t in self.tools.values()]

    def dispatch(self, name: str, arguments: dict) -> dict:
        if name not in self.tools:
            return {"error": f"unknown tool {name!r}"}
        return self.tools[name].run(arguments)


def build_registry(data_ctx: DataContext, treatment_info: Path, cache_dir: Path,
                   literature_backend: str = "pubmed", figures_dir=None,
                   figures_rel_prefix: str = "figures", compound_result=None) -> Registry:
    cache = DiskCache(cache_dir)
    tools = [
        drug_context.make_tool(treatment_info),
        internal_assoc.make_tool(data_ctx),
        depmap.make_tool(data_ctx),
        stringdb.make_tool(cache),
        opentargets.make_tool(cache),
        cbioportal.make_tool(cache),
        pathways.make_tool(cache),
        literature.make_tool(cache, backend=literature_backend),
    ]
    if figures_dir is not None and compound_result is not None:
        tools += figures_mod.make_figure_tools(
            figures_dir=figures_dir, rel_prefix=figures_rel_prefix,
            data_ctx=data_ctx, compound_result=compound_result, cache=cache,
        )
    return Registry(tools={t.name: t for t in tools})
