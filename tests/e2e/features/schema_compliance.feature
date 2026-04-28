Feature: Query endpoint schema compliance
  The /analyze, /execute, and /verify endpoints return JSON
  conforming to the operator's output schemas across all providers.

  Verifies: .ai/spec/query-api.md

  Scenario: /analyze response validates against analysis schema
    Given a running agent service
    And the analysis output schema
    When I send an analysis query with the output schema
    Then the response status is 200
    And the response body validates against the analysis schema
    And the response contains "options" as a non-empty array
    And each option has required fields "diagnosis" "proposal" "components"

  Scenario: /execute response validates against execution schema
    Given a running agent service
    And the execution output schema
    And an execution context with an approved option
    When I send an execution query with the output schema and context
    Then the response status is 200
    And the response body validates against the execution schema
    And the response contains "actionsTaken" as an array
    And the response contains "verification" with boolean "conditionImproved"

  Scenario: /verify response validates against verification schema
    Given a running agent service
    And the verification output schema
    And a verification context from a previous execution
    When I send a verification query with the output schema and context
    Then the response status is 200
    And the response body validates against the verification schema
    And the response contains "conditionImproved" as a boolean

  Scenario: Schema fields survive QueryResponse wrapping
    Given a running agent service
    And the analysis output schema
    When I send an analysis query with the output schema
    Then the response contains wrapper field "success"
    And the response contains wrapper field "summary"
    And all analysis schema fields are present in the response
