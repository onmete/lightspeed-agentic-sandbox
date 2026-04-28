"""BDD test module for skill invocation scenarios.

Wires features/skill_invocation.feature to step definitions.
"""

from pytest_bdd import scenarios

from .steps.given import *
from .steps.then import *
from .steps.when import *

scenarios("features/skill_invocation.feature")
