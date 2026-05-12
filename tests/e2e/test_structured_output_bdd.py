"""pytest-bdd glue for structured output E2E scenarios."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_bdd import scenarios

pytestmark = pytest.mark.e2e

_FEATURE = Path(__file__).parent / "features" / "structured_output.feature"
scenarios(_FEATURE.as_posix())
