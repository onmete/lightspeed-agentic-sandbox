"""BDD test module for schema compliance scenarios.

Wires features/schema_compliance.feature to step definitions.
"""

from pytest_bdd import scenarios

from .steps.given import *
from .steps.then import *
from .steps.when import *

scenarios("features/schema_compliance.feature")
