"""Every bioclip_models submodule must import cleanly.

Catches broken imports (bad circular refs, missing optional-dep guards) early
without the weight of exercising the actual functionality. Replaces the
inline `python -c "from bioclip_models import ..."` step that the CI ran
before pytest was available.
"""

from __future__ import annotations

import importlib

import pytest

_SUBMODULES = [
    "bundle",
    "cli",
    "eval",
    "export",
    "gbif",
    "geo",
    "inat",
    "schema",
    "verify",
]


@pytest.mark.parametrize("name", _SUBMODULES)
def test_submodule_imports(name: str) -> None:
    importlib.import_module(f"bioclip_models.{name}")
