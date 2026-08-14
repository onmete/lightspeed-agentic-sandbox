Feature: Reasoning configuration via LIGHTSPEED_REASONING_CONFIG
  Verifies: .ai/spec/what/configuration.md (rule 9a)
  Verifies: .ai/spec/what/provider-contract.md (rules 18–21)
  Harness: .ai/spec/what/e2e-testing.md
  Requires: LIGHTSPEED_REASONING_CONFIG env var set on the sandbox process.
  Skipped when LIGHTSPEED_REASONING_CONFIG is not configured.

  Scenario: Run succeeds with reasoning configured
    Given the sandbox service is running with reasoning configured
    When I POST run with a simple reasoning query
    Then the HTTP response status code is 200
    And success is true
    And the response has a non-empty summary
    And the response summary contains the reasoning answer

  Scenario: Reasoning does not break structured output
    Given the sandbox service is running with reasoning configured
    And a flat output schema with required fields has been prepared
    When I POST run with the prepared schema and query
    Then the HTTP response status code is 200
    And success is true
    And the response JSON validates against the output schema
