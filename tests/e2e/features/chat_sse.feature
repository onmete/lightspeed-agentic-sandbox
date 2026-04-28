Feature: Chat SSE event stream contract
  The /chat endpoint streams Server-Sent Events with a defined
  structure matching the SSE protocol in the spec.

  Verifies: .ai/spec/chat-api.md

  Scenario: Chat stream has correct event ordering
    Given a running agent service
    When I send a chat message
    Then the stream starts with a "status" event
    And the stream contains at least one "text" event with content
    And the stream ends with a "done" event
    And the "done" event contains "conversationId"

  Scenario: Conversation continuity
    Given a running agent service
    When I send a chat message and receive a conversation ID
    And I send a follow-up message with that conversation ID
    Then the follow-up stream ends with a "done" event
    And the follow-up "done" event contains the same conversation ID

  Scenario: Tool call events have required fields
    Given a running agent service
    When I send a chat message that triggers tool use
    And the stream contains "tool_call" events
    Then each "tool_call" event has "name" and "input" fields
    And each "tool_result" event has an "output" field
