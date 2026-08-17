Feature: MCP server connectivity
  Verifies: OLS-3443 / OLS-3445 (MCP server connectivity for agentic sandbox)
  The sandbox parses LIGHTSPEED_MCP_SERVERS, resolves credentials, connects
  to configured MCP servers, and can invoke MCP-provided tools during a run.

  Background:
    Given the sandbox service is running with MCP servers configured

  Scenario: Sandbox starts successfully with MCP servers configured
    When I GET /health
    Then the HTTP response status code is 200
    And the response body status is ok

  Scenario: Agent can invoke an MCP tool and use its output
    Given an MCP tool invocation query has been prepared
    When I POST run with the prepared schema and query
    Then the HTTP response status code is 200
    And success is true
    And the response summary contains the sentinel namespace from the tool

  Scenario: Agent returns a graceful error envelope when a tool call fails
    Given an MCP query targeting a nonexistent tool has been prepared
    When I POST run with the prepared schema and query
    Then the HTTP response status code is 200
    And success is false
    And the response has a non-empty summary
