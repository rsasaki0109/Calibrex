"""Visualization helpers, imported when their public functions are requested."""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from calibrex.visualization.comparison_table import (
        render_comparison_table as render_comparison_table,
    )
    from calibrex.visualization.comparison_table import (
        write_comparison_table as write_comparison_table,
    )
    from calibrex.visualization.evidence_card import (
        render_evidence_card as render_evidence_card,
    )
    from calibrex.visualization.evidence_card import (
        write_evidence_card as write_evidence_card,
    )
    from calibrex.visualization.report import render_html_report as render_html_report

__all__ = [
    "render_comparison_table",
    "render_evidence_card",
    "render_html_report",
    "write_comparison_table",
    "write_evidence_card",
]

_EXPORT_MODULES = {
    "render_comparison_table": "calibrex.visualization.comparison_table",
    "write_comparison_table": "calibrex.visualization.comparison_table",
    "render_evidence_card": "calibrex.visualization.evidence_card",
    "write_evidence_card": "calibrex.visualization.evidence_card",
    "render_html_report": "calibrex.visualization.report",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value: object = getattr(_import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
