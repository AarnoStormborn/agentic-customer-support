/**
 * Runtime tests — the sources enricher (pure logic extracted from the session
 * wrapper) plus the guardrail-blocked no-op contract.
 */
import { describe, it, expect } from "vitest";
import { createSourceEnricher, sourceKey, type SourceLike } from "../src/runtime/sources.js";
import { MAX_DONE_SOURCES } from "../src/config/limits.js";

function toolEnd(sources: SourceLike[]) {
  return {
    type: "tool_execution_end",
    toolName: "kb_search",
    result: { details: { sources } },
  };
}

function settled() {
  return { type: "agent_settled", messages: [] };
}

describe("createSourceEnricher", () => {
  it("attaches tool sources to agent_settled and passes other events through", () => {
    let attached: SourceLike[] | null = null;
    const enricher = createSourceEnricher((sources) => {
      attached = sources;
      return { type: "agent_settled", sources };
    });

    const start = { type: "agent_start" };
    expect(enricher.handle(start)).toBe(start);

    enricher.handle(toolEnd([{ type: "kb", title: "lg.pdf" }]));
    enricher.handle(toolEnd([{ type: "sql", title: "ticket #1", row: { ticket_id: 1 } }]));

    const out = enricher.handle(settled());
    expect(attached).toHaveLength(2);
    expect(out).toMatchObject({ type: "agent_settled", sources: attached });
  });

  it("dedupes identical sources across multiple tool results", () => {
    let attached: SourceLike[] | null = null;
    const enricher = createSourceEnricher((s) => {
      attached = s;
      return { type: "agent_settled", sources: s };
    });
    enricher.handle({ type: "agent_start" });
    enricher.handle(toolEnd([{ type: "kb", title: "lg.pdf" }]));
    enricher.handle(toolEnd([{ type: "kb", title: "lg.pdf" }])); // duplicate
    enricher.handle(toolEnd([{ type: "sql", title: "ticket #1", row: { ticket_id: 1 } }]));
    enricher.handle(settled());
    expect(attached).toHaveLength(2);
  });

  it("caps sources at MAX_DONE_SOURCES", () => {
    let attached: SourceLike[] | null = null;
    const enricher = createSourceEnricher((s) => {
      attached = s;
      return { type: "agent_settled", sources: s };
    });
    enricher.handle({ type: "agent_start" });
    const many = Array.from({ length: MAX_DONE_SOURCES + 10 }, (_, i) => ({
      type: "sql",
      title: `ticket #${i}`,
      row: { ticket_id: i },
    }));
    enricher.handle(toolEnd(many));
    enricher.handle(settled());
    expect(attached).toHaveLength(MAX_DONE_SOURCES);
  });

  it("resets collection on a new agent_start", () => {
    let attached: SourceLike[] | null = null;
    const enricher = createSourceEnricher((s) => {
      attached = s;
      return { type: "agent_settled", sources: s };
    });
    enricher.handle({ type: "agent_start" });
    enricher.handle(toolEnd([{ type: "kb", title: "a.pdf" }]));
    enricher.handle(settled());
    expect(attached).toHaveLength(1);

    enricher.handle({ type: "agent_start" });
    enricher.handle(settled());
    expect(attached).toHaveLength(0);
  });

  it("sourceKey gives stable identity", () => {
    expect(sourceKey({ type: "kb", title: "lg.pdf" })).toBe("kb:lg.pdf:");
  });
});
