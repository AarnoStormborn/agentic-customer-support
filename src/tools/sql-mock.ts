/**
 * Mock SQL executor for the tickets_query tool — a tiny SELECT interpreter over
 * an in-memory tickets table, so the tool is fully exercisable without Postgres.
 *
 * Real mode (SQL_MODE=real + SQL_IMPL pointing at a module that exports
 * getPool()) runs against Postgres with statement_timeout + read-only
 * transaction; see sql-tool.ts.
 */

import { MAX_SQL_RESULT_ROWS } from "../config/limits.js";

export interface TicketRow {
  id: number;
  customer_name: string;
  product: string;
  issue: string;
  status: string;
  priority: string;
  created_at: string;
}

export const MOCK_TICKETS: TicketRow[] = [
  { id: 10231, customer_name: "A. Rivera", product: "LG OLED TV 65C4", issue: "TV keeps disconnecting from home Wi-Fi", status: "open", priority: "high", created_at: "2026-07-28" },
  { id: 10217, customer_name: "J. Chen", product: "LG Soundbar S80QR", issue: "No sound over eARC", status: "in_progress", priority: "medium", created_at: "2026-07-25" },
  { id: 10198, customer_name: "M. Okafor", product: "Samsung Refrigerator", issue: "Ice maker stopped working after filter change", status: "open", priority: "medium", created_at: "2026-07-21" },
  { id: 10154, customer_name: "S. Patel", product: "Whirlpool Washer", issue: "F5 E2 door latch error", status: "resolved", priority: "low", created_at: "2026-07-14" },
  { id: 10122, customer_name: "D. Kim", product: "LG OLED TV 55C4", issue: "Remote control not pairing", status: "closed", priority: "low", created_at: "2026-07-02" },
];

const KEYWORDS = ["select", "from", "where", "and", "or", "order by", "limit", "as", "group by", "having"];

interface ParsedQuery {
  columns: string[]; // "*" or list of bare column names
  where: string | null; // raw WHERE text (parsed lazily into an expression)
  orderBy: string | null;
  orderDir: "asc" | "desc";
  limit: number | null;
}

/**
 * Parse the tiny supported subset:
 *   SELECT col1, col2 | * FROM tickets [WHERE cond [AND|OR cond ...]] [ORDER BY col [ASC|DESC]] [LIMIT n]
 * Conditions: col = 'value' | col != 'value' | col ILIKE '%v%' | col IN (...)
 */
export function parseSelect(query: string): ParsedQuery | { error: string } {
  const q = query.trim().replace(/;+$/, "");
  const fromIdx = q.search(/\bfrom\b/i);
  if (fromIdx < 0) return { error: "missing FROM clause" };
  const selectPart = q.slice(0, fromIdx);
  let rest = q.slice(fromIdx + 4);
  // strip table name + optional alias (ident, dotted ident, or quoted) up to WHERE/ORDER/LIMIT/;
  rest = rest.replace(
    /^\s*(?:`[^`]*`|"[^"]*"|[A-Za-z_][\w.]*)(?:\s+(?:`[^`]*`|"[^"]*"|[A-Za-z_][\w.]*))?(?=\s*(?:where|order|limit|;|$))/i,
    "",
  );

  const colsRaw = selectPart.replace(/^\s*select\b/i, "").trim();
  if (!colsRaw) return { error: "no columns selected" };
  const columns = colsRaw === "*" ? ["*"] : colsRaw.split(",").map((c) => c.trim()).filter(Boolean);

  const orderIdx = rest.search(/\border by\b/i);
  const limitIdx = rest.search(/\blimit\b/i);
  let wherePart = rest;
  let orderBy: string | null = null;
  let orderDir: "asc" | "desc" = "asc";
  let limit: number | null = null;

  if (limitIdx >= 0) {
    const after = rest.slice(limitIdx + 5).trim();
    const n = Number.parseInt(after, 10);
    if (Number.isNaN(n) || n < 0) return { error: "invalid LIMIT" };
    limit = n;
    wherePart = rest.slice(0, limitIdx);
  }
  if (orderIdx >= 0) {
    const after = rest.slice(orderIdx + 8).trim().split(/\s+/);
    orderBy = after[0] ?? null;
    if (!orderBy) return { error: "ORDER BY needs a column" };
    if (/^desc$/i.test(after[1] ?? "")) orderDir = "desc";
    wherePart = rest.slice(0, orderIdx);
  }

  const where: string | null = (() => {
    let w = wherePart.replace(/^\s*where\b/i, "").trim();
    return w || null;
  })();

  return { columns, where, orderBy, orderDir, limit };
}

/** Normalize hyphens so "Wi-Fi" matches a search for "wifi". */
function normalizeForMatch(s: string): string {
  return s.toLowerCase().replace(/[-–—]/g, "");
}

/**
 * Mock ILIKE: split the pattern into whitespace-separated tokens (strip %)
 * and require each token to appear in the cell (hyphen-normalized).
 * This is deliberately looser than real SQL — "%lg tv%" matches "LG OLED TV",
 * "%wifi%" matches "Wi-Fi" — so the demo specialist queries succeed.
 */
function likeMatch(value: string, cell: string): boolean {
  const tokens = normalizeForMatch(value)
    .replace(/%/g, " ")
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0);
  if (tokens.length === 0) return true;
  const cellNorm = normalizeForMatch(cell);
  return tokens.every((t) => cellNorm.includes(t));
}

// --- tiny boolean WHERE evaluator (AND binds tighter than OR, parens supported) ---

type Expr = { kind: "or"; parts: Expr[] } | { kind: "and"; parts: Expr[] } | { kind: "cond"; cond: string };

class WhereParser {
  private pos = 0;
  constructor(private readonly src: string) {}

  parse(): Expr {
    const e = this.parseOr();
    this.skipWs();
    if (this.pos < this.src.length) {
      throw new Error(`unexpected token at '${this.src.slice(this.pos)}'`);
    }
    return e;
  }

  private parseOr(): Expr {
    const parts = [this.parseAnd()];
    for (;;) {
      this.skipWs();
      const m = this.src.slice(this.pos).match(/^or\b/i);
      if (!m) break;
      this.pos += m[0].length;
      parts.push(this.parseAnd());
    }
    return parts.length === 1 ? parts[0]! : { kind: "or", parts };
  }

  private parseAnd(): Expr {
    const parts = [this.parsePrimary()];
    for (;;) {
      this.skipWs();
      const m = this.src.slice(this.pos).match(/^and\b/i);
      if (!m) break;
      this.pos += m[0].length;
      parts.push(this.parsePrimary());
    }
    return parts.length === 1 ? parts[0]! : { kind: "and", parts };
  }

  private parsePrimary(): Expr {
    this.skipWs();
    if (this.src[this.pos] === "(") {
      this.pos += 1;
      const inner = this.parseOr();
      this.skipWs();
      if (this.src[this.pos] !== ")") throw new Error("unbalanced parentheses in WHERE");
      this.pos += 1;
      return inner;
    }
    const start = this.pos;
    // scan to the next top-level AND/OR or ')' — skipping quoted literals
    let i = start;
    while (i < this.src.length) {
      const ch = this.src[i]!;
      if (ch === "'") {
        i += 1;
        while (i < this.src.length && this.src[i] !== "'") i += 1;
        i += 1;
        continue;
      }
      if (ch === "(") { i += 1; continue; }
      if (ch === ")") break;
      if (/\s/.test(ch)) {
        const m = this.src.slice(i).match(/^\s+(and|or)\b/i);
        if (m) break;
      }
      i += 1;
    }
    this.pos = i;
    const cond = this.src.slice(start, i).trim();
    if (!cond) throw new Error("empty condition in WHERE");
    return { kind: "cond", cond };
  }

  private skipWs(): void {
    while (this.pos < this.src.length && /\s/.test(this.src[this.pos]!)) this.pos += 1;
  }
}

function evalExpr(expr: Expr, row: TicketRow): boolean {
  switch (expr.kind) {
    case "or": return expr.parts.some((p) => evalExpr(p, row));
    case "and": return expr.parts.every((p) => evalExpr(p, row));
    case "cond": return matchesCondition(row, expr.cond);
  }
}

function matchesCondition(row: TicketRow, cond: string): boolean {
  // col = 'value' / col != 'value' / col ILIKE '%v%' / col IN ('a','b')
  const inMatch = cond.match(/^(\w+)\s+in\s*\(([^)]+)\)$/i);
  if (inMatch) {
    const col = inMatch[1]!.toLowerCase();
    const values = (inMatch[2] ?? "").split(",").map((v) => v.trim().replace(/^'|'$/g, "")).filter(Boolean);
    const cell = normalizeForMatch(String((row as unknown as Record<string, unknown>)[col] ?? ""));
    return values.some((v) => cell === normalizeForMatch(v));
  }
  const opMatch = cond.match(/^(\w+)\s*(=|!=|<>|>=|<=|>|<|ilike|like)\s*(.+)$/i);
  if (!opMatch) return true; // unparseable condition: ignore rather than fail
  const col = opMatch[1]!.toLowerCase();
  const op = opMatch[2]!.toLowerCase();
  let value = (opMatch[3] ?? "").trim().replace(/^'|'$/g, "");
  const rawCell = String((row as unknown as Record<string, unknown>)[col] ?? "");
  switch (op) {
    case "=": return normalizeForMatch(rawCell) === normalizeForMatch(value);
    case "!=":
    case "<>": return normalizeForMatch(rawCell) !== normalizeForMatch(value);
    case ">": return Number(rawCell) > Number(value);
    case "<": return Number(rawCell) < Number(value);
    case ">=": return Number(rawCell) >= Number(value);
    case "<=": return Number(rawCell) <= Number(value);
    case "like":
    case "ilike": return likeMatch(value, rawCell);
    default: return true;
  }
}

export interface MockQueryResult {
  columns: string[];
  rows: Record<string, unknown>[];
  truncated: boolean;
}

/** Execute a SELECT against the in-memory ticket table. */
export function executeMockQuery(query: string): MockQueryResult {
  const parsed = parseSelect(query);
  if ("error" in parsed) throw new Error(`mock SQL parse error: ${parsed.error}`);
  const { columns, where, orderBy, orderDir, limit } = parsed;

  let rows: TicketRow[];
  if (where) {
    const expr = new WhereParser(where).parse();
    rows = MOCK_TICKETS.filter((r) => evalExpr(expr, r));
  } else {
    rows = [...MOCK_TICKETS];
  }

  if (orderBy) {
    const col = orderBy.toLowerCase() as keyof TicketRow;
    rows = [...rows].sort((a, b) => {
      const av = a[col] ?? "";
      const bv = b[col] ?? "";
      const cmp = typeof av === "number" && typeof bv === "number"
        ? av - bv
        : String(av).localeCompare(String(bv));
      return orderDir === "desc" ? -cmp : cmp;
    });
  }
  const max = Math.min(limit ?? MAX_SQL_RESULT_ROWS, MAX_SQL_RESULT_ROWS);

  // COUNT(*) / COUNT(1) — the one aggregate the mock understands
  if (columns.length === 1 && /^count\s*\(\s*(\*|1)\s*\)$/i.test(columns[0] ?? "")) {
    const total = rows.length;
    const finalCount = Math.min(total, MAX_SQL_RESULT_ROWS);
    return {
      columns: ["count"],
      rows: [{ count: total }],
      truncated: total > finalCount,
    };
  }

  const truncated = rows.length > max;
  rows = rows.slice(0, max);

  const outCols = columns[0] === "*" ? Object.keys(MOCK_TICKETS[0] ?? {}) : columns;
  const outRows = rows.map((r) => {
    const rec: Record<string, unknown> = {};
    for (const c of outCols) rec[c] = (r as unknown as Record<string, unknown>)[c] ?? null;
    return rec;
  });

  return { columns: outCols, rows: outRows, truncated };
}

export { KEYWORDS };
