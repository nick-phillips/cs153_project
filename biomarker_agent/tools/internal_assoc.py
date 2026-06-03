"""Tool: associate a feature with the compound's response in the training data."""

from ..datactx import DataContext
from .base import Tool

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "feature_name": {"type": "string", "description": "Full feature name, e.g. 'GE_ITGA1'"},
        "compound_id": {"type": "string", "description": "BRD id of the compound"},
    },
    "required": ["feature_name", "compound_id"],
}


def make_tool(ctx: DataContext) -> Tool:
    return Tool(
        name="internal_association",
        description=(
            "Quantify how a feature relates to this compound's response across cell lines "
            "in the actual training data: Pearson/Spearman correlation, differential activity "
            "(high vs low feature tertile), and direction. Use to confirm a selected feature "
            "tracks response and in which direction."
        ),
        input_schema=INPUT_SCHEMA,
        handler=lambda feature_name, compound_id: ctx.associate(feature_name, compound_id),
    )
