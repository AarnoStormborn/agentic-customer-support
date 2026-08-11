# CONTRACT-NOTES — deviations & integration notes (agent-runtime track)

Notes for the orchestrator and the api-streaming track. The public exports
match `docs/design/integration-contract.md`; everything here is additive.

## Export surface (contract-compliant)

```ts
// src/runtime/index.ts
createSupportRuntime(opts?: { model?, chatId?, sessionDir? }): Promise<SupportRuntime>
type SupportRuntime = { prompt, steer, abort, subscribe, getLastMessages, dispose }
// src/guardrails/extension.ts
guardrailsExtension(pi: unknown): void        // exact contract signature
supportGuardrails: InlineExtension             // named extension used by the runtime loader
```

- `SupportRuntimeImpl` (the concrete class) is exported from
  `src/runtime/session.ts` too — the chat CLI uses it. It adds one method beyond
  the contract: `promptWithBudget(text, budgetMs?)` (per-turn timeout + abort).
- `subscribe` emits **raw pi SDK `AgentSessionEvent`s** — agent-runtime emits no
  SSE; api-streaming's bridge owns the SDK→SSE mapping (design §3.4).

## Tool names (the four supervisor tools)

`kb_search` (rag) · `tickets_query` (sql) · `web_search` (web) · `route_to_agent`.
`route_to_agent` accepts `{ agent: "rag"|"sql"|"web", query }` and returns
`content` (specialist answer) + `details: { sources, childToolCalls, model,
turnCount, tokens }`. Guardrail hook validates `agent` before spawn.

## Decisions worth knowing

1. **`guardrailsExtension(pi: unknown)`** — internally casts to `ExtensionAPI`
   and shares the same factory as `supportGuardrails`. Kept `unknown` exactly as
   the contract states so api-streaming can import it without SDK types.
2. **`model?`/`PI_MODEL` matching is lenient** — accepts `"provider/id"` or a bare
   id (`"deepseek-v4-flash"` → `opencode-go/deepseek-v4-flash`); ambiguous bare
   ids throw. Falls back to a preferred list, then first available.
3. **Child sessions reuse the parent's `ModelRuntime`** — `createSupportRuntime`
   calls `configureRouteToAgent({ modelRuntime, supervisorModel })`; the
   `route_to_agent` tool lazily creates one only if used standalone.
4. **Context-hook safety note is a user-role message** — the SDK's in-session
   `Message` union has no `"system"` role (`UserMessage | AssistantMessage |
   ToolResultMessage`), so the note is prepended as a clearly-marked user
   message rather than a system message.
5. **Mock SQL is deliberately looser than real SQL** — `ILIKE '%lg tv%'` matches
   per-token ("LG OLED TV"), `'%wifi%'` matches "Wi-Fi" (hyphen-normalized).
   Real mode (`SQL_MODE=real`) runs real Postgres semantics; see DEPS.md.
6. **`sessionDir`** → `SessionManager.create(dir)` (JSONL); default in-memory.
7. **SDK event nuance** — session-level `turn_start` carries **no** `turnIndex`
   (only the extension-level `TurnStartEvent` does). Bridges mapping `turn_start`
   should not rely on `turnIndex` from `subscribe()` events.

## Untouched (owned by other tracks)

`src/db/`, `src/retrieval/` (real impl), `src/server/`, `src/streaming/`,
`src/queue/`, `src/mcp/`, `schema.sql`, provision scripts.
