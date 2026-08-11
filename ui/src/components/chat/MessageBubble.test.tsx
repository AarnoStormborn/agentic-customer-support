/**
 * MessageBubble.test.tsx — component test: markdown rendering, streaming caret,
 * sources, error state (happy-dom, no network).
 */
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MessageBubble } from "./MessageBubble";
import type { ChatMessage } from "../../stores/chatStore";

function assistant(overrides: Partial<ChatMessage> = {}): ChatMessage {
  return {
    id: "a1",
    role: "assistant",
    text: "",
    status: "streaming",
    createdAt: Date.now(),
    ...overrides,
  };
}

describe("MessageBubble", () => {
  it("renders user messages right-aligned with the text", () => {
    render(
      <MessageBubble
        message={{
          id: "u1",
          role: "user",
          text: "help me",
          status: "sent",
          createdAt: Date.now(),
        }}
      />,
    );
    expect(screen.getByText("help me")).toBeTruthy();
  });

  it("renders assistant markdown (bold + inline code) as HTML", () => {
    render(
      <MessageBubble
        message={assistant({ text: "Press **OK** then type `admin`.", status: "done" })}
      />,
    );
    expect(screen.getByText("OK").tagName).toBe("STRONG");
    expect(screen.getByText("admin").tagName).toBe("CODE");
  });

  it("shows the streaming caret while tokens arrive", () => {
    render(<MessageBubble message={assistant({ text: "Working" })} />);
    expect(screen.getByLabelText("streaming")).toBeTruthy();
  });

  it("renders sources with the cited ticket", () => {
    render(
      <MessageBubble
        message={assistant({
          text: "Done",
          status: "done",
          sources: [
            { type: "sql", title: "ticket #42", row: { ticket_id: 42, ticket_subject: "WiFi" } },
            { type: "kb", title: "lg-manual.pdf", sectionPath: "4.2 Wi-Fi", score: 0.91 },
          ],
        })}
      />,
    );
    expect(screen.getByText("Sources")).toBeTruthy();
    expect(screen.getByText("ticket #42")).toBeTruthy();
    expect(screen.getByText("lg-manual.pdf")).toBeTruthy();
    expect(screen.getAllByTestId("source-card")).toHaveLength(2);
  });

  it("keeps partial text + shows an error panel on stream error", () => {
    render(
      <MessageBubble
        message={assistant({
          text: "Partial answer",
          status: "error",
          error: { code: "provider_error", message: "The provider timed out", retryable: true },
        })}
      />,
    );
    expect(screen.getByText("Partial answer")).toBeTruthy();
    expect(screen.getByText("Stream stopped")).toBeTruthy();
    expect(screen.getByText("The provider timed out")).toBeTruthy();
  });

  it("tags cancelled bubbles without a caret", () => {
    render(<MessageBubble message={assistant({ text: "Half", status: "cancelled" })} />);
    expect(screen.getByText("Cancelled")).toBeTruthy();
    expect(screen.queryByLabelText("streaming")).toBeNull();
  });
});
