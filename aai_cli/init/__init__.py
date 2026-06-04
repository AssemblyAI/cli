# aai_cli/init/__init__.py
from __future__ import annotations

from aai_cli.init.keys import resolve_optional_api_key
from aai_cli.init.scaffold import scaffold as scaffold_fn
from aai_cli.init.scaffold import target_conflict
from aai_cli.init.templates import TEMPLATE_ORDER, TEMPLATES, is_template, title_for

__all__ = [
    "TEMPLATES",
    "TEMPLATE_ORDER",
    "is_template",
    "resolve_optional_api_key",
    "scaffold_fn",
    "target_conflict",
    "title_for",
]
