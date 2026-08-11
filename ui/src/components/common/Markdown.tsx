/**
 * common/Markdown.tsx — react-markdown + remark-gfm with app styling.
 *
 * Inline citation markers `[1]` `[2]` (ui.md §2.2) are rendered as styled
 * superscripts; clicking one scrolls to the message's sources list.
 */
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { JSX } from "react";

/** Render [n] citation markers inside a text node as accent superscripts. */
function withCitations(text: string, onCitationClick?: (n: number) => void) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const m = /^\[(\d+)\]$/.exec(part);
    if (!m) return part;
    const n = Number(m[1]);
    return (
      <sup
        key={i}
        onClick={onCitationClick ? () => onCitationClick(n) : undefined}
        className={`text-accent ${onCitationClick ? "cursor-pointer" : ""}`}
        title={onCitationClick ? `Source [${n}]` : undefined}
      >
        [{n}]
      </sup>
    );
  });
}

const textComponents: Array<keyof JSX.IntrinsicElements> = ["p", "li", "td"] as const;

export function Markdown({
  children,
  onCitationClick,
}: {
  children: string;
  onCitationClick?: (n: number) => void;
}) {
  const components: Components = {
    a: ({ href, children: c }) => (
      <a href={href} target="_blank" rel="noreferrer noopener">
        {c}
      </a>
    ),
    code: ({ className, children: c, ...props }) => {
      const inline = !className;
      return inline ? (
        <code {...props}>{c}</code>
      ) : (
        <code className={className} {...props}>
          {c}
        </code>
      );
    },
  };
  for (const tag of textComponents) {
    (components as Record<string, unknown>)[tag] = ({ children: c }: { children?: unknown }) => {
      if (typeof c === "string") return <>{withCitations(c, onCitationClick)}</>;
      return <>{c}</>;
    };
  }

  return (
    <div className="md-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {children}
      </ReactMarkdown>
    </div>
  );
}
