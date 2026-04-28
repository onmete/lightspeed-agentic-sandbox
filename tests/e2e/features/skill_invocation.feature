Feature: Skill discovery and tool execution
  The agent discovers skills from the workspace and executes
  tool scripts, producing verifiable structured output.

  Verifies: .ai/spec/provider-contract.md

  Scenario: Find-token skill produces verifiable tokens
    Given a running agent service
    And a workspace with the find-token skill
    When I ask the agent to find the hidden token
    Then the agent response validates against the analysis schema
    And the hidden token file exists in the workspace output
    And the response contains both tokens from the hidden token file
