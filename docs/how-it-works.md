# How the Streaming Agent Loop Works (first principles → code)

> Written for anyone (re)building this pattern in another application.
> Reference implementation: `src/streaming/bridge.ts`, `src/runtime/session.ts`,
> `src/server/routes/chat.ts`, `ui/src/stores/chatStore.ts`, `ui/src/lib/sse.ts`.

---

## 1. First principles — what is *actually* happening

Everything you see in the UI reduces to three facts:

### Fact 1: An LLM is a stateless token generator over HTTP
"Chat" does not exist. Every response is a fresh HTTP call where you send the
**entire conversation history** plus the new question. The model has no memory —
the messages array *is* the memory.

### Fact 2: "Streaming" = chunked HTTP
The provider doesn't return one big JSON. It returns an HTTP response whose body
is a stream of small chunks, each carrying a few tokens. You read the stream and
forward each delta as it arrives. That is the entire secret behind "live typing."

### Fact 3: Tool calling is just structured text
A tool call is not magic — the model emits a special JSON token:
`call kb_search(query="lg tv wifi reset")`. **You** (the app) decide what to do:
run the function, get the result, and append it to the conversation as a new
message. Then you call the model again — now it can *see* the tool result and
answer.

### The agent loop

```
messages = [user: "reset wifi on lg tv"]
loop:
  stream = LLM(messages + tools)                # Facts 1 + 2
  for each chunk in stream:
    if text token:  forward it to the client    # streaming
    if tool_call:   remember it, stop reading   # Fact 3
  if no tool_call: break                        # model answered → done
  result = execute(tool_call)                   # YOUR code (DB / vector / web)
  messages.append(tool result)
  loop                                          # model reasons over the result
```

The observed pattern — *responds, then calls tools, then streams the answer* —
is just this loop with streaming at every layer:

```
Turn 1: "Let me look that up…" + tool_call   → (pause: tool runs, shown in UI)
Turn 2: final answer, streamed token-by-token
```

---

## 2. A real event trace (captured from this system)

```
event: turn_start                              ← LLM turn 1 begins
event: token  delta:"Let me"                   ← preamble streams…
event: token  delta:" look that up…"           ← …491 token events total
event: tool_start  {"toolName":"kb_search","args":{"query":"LG TV wifi reset"}}
event: tool_end    {"toolName":"kb_search","isError":false}
event: tool_start  {"toolName":"route_to_agent","args":{"agent":"rag","query":"…"}}
event: tool_end    {"toolName":"route_to_agent","isError":false}
event: turn_end                               ← turn 1 ends (tool results appended)
event: turn_start                              ← LLM turn 2 begins
event: token …  token …                       ← final answer streams
event: done  {"message":"…","sources":[…]}     ← authoritative text + citations
```

Note the **two** `tool_start`s: the supervisor called `kb_search` *and*
`route_to_agent` for the same question — the double-retrieval behaviour (see
`docs/plan.md` "open items").

---

## 3. Backend POV (this repo)

| Concern | This repo | Generic version |
|---|---|---|
| Run the loop | `createAgentSession()` (pi SDK) | any agent framework, or the hand-rolled loop in §1 |
| Deltas + tool events | `session.subscribe(events)` | provider SDK's stream iterator |
| Map to a wire protocol | `src/streaming/bridge.ts` — pure 1:1 mapper | your own `token`/`tool`/`done` event names |
| Deliver over HTTP | SSE via `@fastify/sse` (one long-lived `text/event-stream`) | any HTTP server |
| Reconnect safety | ring buffer (last 200 events) + `Last-Event-ID` replay | optional, cheap |
| Kill a turn | `POST /cancel` → `session.abort()` | abort the in-flight stream + tool execution |

### SSE vs WebSocket (one line each)
- **SSE** — one-way server→client stream over plain HTTP; the browser
  `EventSource` auto-reconnects. Perfect for token deltas.
- **WebSocket** — two-way; needed only if the *client* must push mid-turn
  ("stop", "steer"). This repo uses REST for steer/cancel and skips WS.

### The bridge is a pure mapper
`attachBridge(session, sink, ctx)` subscribes to SDK events and emits SSE frames:

```ts
case "message_update": {
  const m = event.assistantMessageEvent;
  if (m.type === "text_delta")    sink.emit("token", { delta: m.delta });
  if (m.type === "thinking_delta") sink.emit("thinking", { delta: m.delta });
}
case "tool_execution_start": sink.emit("tool_start", { toolName, args });
case "tool_execution_end":   sink.emit("tool_end", { toolName, isError });
case "agent_settled":        sink.emit("done", { message, sources });
```

Key decisions that made this robust:
- **Sources are attached at `agent_settled`** by a wrapper that collects
  `details.sources` from each `tool_execution_end` (deduped, capped — the "second
  turn_start" bug in `docs/lessons.md` §16 taught us to reset only on `agent_start`).
- **`done` carries the authoritative final text** — streamed tokens can be
  partial on reconnect; the client swaps the partial text for the final one.
- **Events buffer in a ring buffer** so a late SSE client replays the turn.

---

## 4. Frontend POV (this repo's React UI)

The UI is a **state machine driven by events**, not by fetching responses:

```
messages: [{role:"user",text}, {role:"assistant",text:"…",streaming:true}]
activity: [{tool:"kb_search", status:"running"}]
sources:  []
```

1. **Send** → optimistically append the user bubble + an empty assistant bubble →
   `POST /api/chat` → open `EventSource(/api/chat/:id/events)`.
2. **`token`** → append `delta` to the open assistant bubble. Batched via
   `requestAnimationFrame` — at most one re-render per frame regardless of how
   fast tokens arrive.
3. **`tool_start` / `tool_end`** → push/update a card in the activity feed
   ("kb_search — running → done"). This makes the tool-execution pause feel
   intentional.
4. **`done`** → replace the partial text with the authoritative text + render
   `sources[]` as citations.
5. **`error`** → keep partial text, mark the bubble errored, offer retry.
6. **Reconnect** → `EventSource` reconnects automatically; the server replays
   missed events via `Last-Event-ID`.

The one rule that makes streaming feel right: **append, never replace** — except
at `done`, where you swap in the final text.

---

## 5. The recipe (for another application)

### Backend — the core loop (~60 lines)

```js
// 1. The loop, with streaming + tools
async function runTurn(conversation, sse) {
  while (true) {
    const stream = await provider.streamChat({ messages: conversation, tools });
    let toolCall = null;
    for await (const chunk of stream) {
      if (chunk.delta)  sse.send({ event: "token", data: { delta: chunk.delta } });
      if (chunk.toolCall) { toolCall = chunk.toolCall; break; }
    }
    if (!toolCall) break;                         // model answered → done
    sse.send({ event: "tool_start", data: { toolName: toolCall.name } });
    const result = await executeTool(toolCall);   // YOUR code: DB / vector / web
    sse.send({ event: "tool_end", data: { toolName: toolCall.name } });
    conversation.push({ role: "tool", content: result });
  }
  sse.send({ event: "done", data: { sources } });
}
```

Then:
- expose it at `GET /api/turns/:id` with `Content-Type: text/event-stream`,
  keep the response open, send periodic `: heartbeat` comments;
- add `POST /api/turns/:id/cancel` that aborts the loop;
- (optional) keep a per-turn ring buffer + honor `Last-Event-ID` on reconnect.

### Frontend — the minimal consumer

```js
const es = new EventSource(`/api/turns/${id}`);
es.addEventListener("token",      e => appendToOpenBubble(JSON.parse(e.data).delta));
es.addEventListener("tool_start", e => addActivityCard(JSON.parse(e.data).toolName));
es.addEventListener("tool_end",   e => markActivityDone(JSON.parse(e.data).toolName));
es.addEventListener("done",       e => finalizeBubble(JSON.parse(e.data)));
es.addEventListener("error",      e => handleStreamError());
```

### What you get for free / what you must decide

- **Free:** chunked streaming (Fact 2), tool execution (Fact 3), the loop (Fact 1).
- **Frameworks:** pi SDK (this repo), OpenAI Agents SDK, LangChain — all wrap this
  exact loop; only the event vocabulary differs.
- **Your real work:** the tools themselves (retrieval, SQL, web) and the *event
  schema* (what `tool_start` carries so the UI can render it nicely).
- **Perf rules:** batch frontend appends with rAF; keep `done`'s text
  authoritative; never send full text per token.

---

## 6. Where this lives in the repo

| Piece | File |
|---|---|
| Agent loop + sessions | `src/runtime/session.ts` (`createSupportRuntime`) |
| Supervisor system prompt (routing rules) | `src/agent/support-prompt.ts` |
| SDK event → SSE mapping | `src/streaming/bridge.ts` |
| SSE + WS route handlers | `src/server/routes/chat.ts`, `src/streaming/sse.ts` |
| Turn registry + ring buffer | `src/streaming/registry.ts` |
| UI reducer (token append, done finalize) | `ui/src/stores/chatStore.ts` (`reduceSseEvent`) |
| UI SSE client | `ui/src/lib/sse.ts`, `ui/src/hooks/useChatStream.ts` |
| SSE event schema (authoritative) | `docs/design/backend-agent-retrieval.md` §2.3 |
