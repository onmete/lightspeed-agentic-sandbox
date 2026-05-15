"""pytest-bdd glue for skills E2E scenarios."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_bdd import scenarios

pytestmark = pytest.mark.e2e

_FEATURE = Path(__file__).parent / "features" / "skills.feature"
scenarios(_FEATURE.as_posix())
