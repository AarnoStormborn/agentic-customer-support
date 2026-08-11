/**
 * Guardrail pattern lists — prompt-injection / attack heuristics and PII scrubbers.
 *
 * Kept in a separate module so tests and the tool layer can reuse them without
 * depending on the extension API.
 */

/** Prompt-injection / jailbreak heuristics (case-insensitive). */
export const ATTACK_PATTERNS: RegExp[] = [
  /\bignore\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|messages?)\b/i,
  /\bignore\s+(everything|all instructions|your instructions|the instructions)\b/i,
  /\bdisregard\s+(all\s+)?(previous|prior|above)?\s*(instructions?|prompts?|rules?)\b/i,
  /\b(you are now|act as|pretend to be)\b[^\n]{0,60}\b(no (rules|restrictions|limits)|unfiltered|without restrictions)\b/i,
  /\b(developer mode|dan mode|jailbreak|jail broken|do anything now)\b/i,
  /\bsystem\s*prompt\b/i, // revealing/overriding the system prompt
  /\b(reveal|show|print|leak)\s+(your|the)\s+(system|internal|base)\s+(prompt|instructions?|directives?)\b/i,
  /\brepeat\s+(the|this|above|everything|all)\s+(words|text|prompt|instructions?)\s+(above|exactly|verbatim|back)\b/i,
  /\b(sudo|admin)\s*mode\b/i,
  /\bnever\s+mind\s+your\s+(previous|prior)\s+(instructions?|prompt)\b/i,
  /\bfrom\s+now\s+on\s+you\s+are\b/i,
  /<system[^>]*>|<\/system>|<tool[^>]*>|<\/tool>/i, // fake XML tags
];

export function findAttackPattern(text: string): string | null {
  for (const re of ATTACK_PATTERNS) {
    const m = text.match(re);
    if (m) return m[0];
  }
  return null;
}

/** PII scrubbers — replace matches with a marker. */
export const PII_PATTERNS: { name: string; re: RegExp }[] = [
  { name: "email", re: /[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/gi },
  { name: "phone", re: /(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/g },
  { name: "ssn", re: /\b\d{3}-\d{2}-\d{4}\b/g },
  { name: "credit-card", re: /\b(?:\d{4}[- ]?){3}\d{4}\b/g },
];

export interface ScrubResult {
  text: string;
  scrubbed: string[]; // pattern names that fired
}

export function scrubPii(text: string): ScrubResult {
  let out = text;
  const scrubbed: string[] = [];
  for (const { name, re } of PII_PATTERNS) {
    if (re.test(out)) {
      out = out.replace(re, `[REDACTED:${name}]`);
      scrubbed.push(name);
    }
  }
  return { text: out, scrubbed };
}
