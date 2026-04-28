"""BDD test module for chat SSE event stream scenarios.

Wires features/chat_sse.feature to step definitions.
"""

from pytest_bdd import scenarios

from .steps.given import *
from .steps.then import *
from .steps.when import *

scenarios("features/chat_sse.feature")
