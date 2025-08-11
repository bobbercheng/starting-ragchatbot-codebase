Refactor @backend/ai_generator.py to support sequential tool calling where AI generator can make up to 2 tool calls in separate API rounds.

Current behavior:
- AI generator makes 1 round tool call -> final response

Desired behavior:
- Each tool call should be a separate API request AI generator can reason about previous results.
- Support for complex queries requiring multiple searches from comparisons, multi-part questions, or when information from courses/lessons is needed.

Example flow:
1. User: "Search for a course that discusses the same topic as lesson 4 for course X"
2. AI generator: get course outline for course X -> gets titles of lesson 4
3. AI generator: uses the tile to search for a course that discusses the same topic -> returns course information
4. Claude: provide complete answer

Requirements:
- Maximum 2 sequential rounds per user query
- Terminate when: (a) 2 rounds completed, (b) AI generator response has no tool_use blocks, or (c) tool call fails.
- Preserve conversation context between rounds
- Handle tool execution errors gracefully
- Has unit test

Notes:
- update the system prompt in @backend/ai_generator.py
- Write test that verify the external behavior (API calls made, tools executed, results returned) rather than internal state details.

Use two parallel subagents to brainstorm possible plans, Do not implement any code.