/**
 * src/retrieval/chunk.ts — structural PDF chunker (design backend-agent-retrieval §4.4,
 * data-management §4.2). Replaces legacy split_text's blind fixed-window slicing.
 *
 * Pipeline:
 *   pdf-parse (v2: `new PDFParse({data})` → `getText()` → per-page text)
 *   → line-level heading detection (numbered headings, known section keywords,
 *     ALL-CAPS / title-case short lines)
 *   → heading tree → section units with breadcrumb `headingPath`
 *   → sections larger than the char cap are split at paragraph boundaries with
 *     ~12% overlap; tiny sections merge into the previous chunk
 *   → chunk = { chunkIndex, text, section, headingPath, pageStart, pageEnd }
 */
import { PDFParse } from "pdf-parse";
import { readFile } from "node:fs/promises";
import { basename } from "node:path";

export interface ParsedPage {
  num: number;
  text: string;
}

export interface ParsedDocument {
  docName: string;
  filePath: string;
  pageCount: number;
  totalChars: number;
  pages: ParsedPage[];
}

export interface DocumentChunk {
  chunkIndex: number;
  text: string;
  section: string | null;
  headingPath: string | null;
  pageStart: number | null;
  pageEnd: number | null;
}

export interface ChunkOptions {
  /** hard cap per chunk (chars); design §4.2: ~512 tokens ≈ 1,500–2,000 chars */
  maxChars?: number;
  /** target size before splitting (chars) */
  targetChars?: number;
  /** overlap fraction between consecutive chunks within a section */
  overlapFraction?: number;
  /** sections smaller than this merge into the previous chunk */
  minChars?: number;
}

const DEFAULT_OPTS: Required<ChunkOptions> = {
  maxChars: 2500,
  targetChars: 1900,
  overlapFraction: 0.12,
  minChars: 180,
};

// Section keywords that start a heading even without a number.
const SECTION_KEYWORDS = /\b(troubleshooting|safety|specifications|specification|faq|warranty|installation|maintenance|introduction|appendix|getting started|quick start|glossary|index)\b/i;
// Heading line shapes: "1.", "1.2", "1.2.3", "4 Setting Up", "SECTION 3: …"
const NUMBERED_HEADING = /^\s*(?:section\s+)?\d+(?:\.\d+)*[\.\)]?\s*[A-Za-z0-9]/;

function isHeading(line: string): boolean {
  const t = line.trim();
  if (!t || t.length > 80) return false;
  if (NUMBERED_HEADING.test(t)) return true;
  if (SECTION_KEYWORDS.test(t) && t.length < 60) return true;
  // Title-case / ALL-CAPS short line with no sentence-ending punctuation
  if (/^[A-Z][A-Z0-9 /&'’\-]{2,60}$/.test(t) && !/[.!?:]$/.test(t)) return true;
  return false;
}

/** Breadcrumb depth for numbered headings: "1.2.3" → 3 (pop stack to that depth). */
function headingDepth(heading: string): number | null {
  const m = heading.trim().match(/^(\d+(?:\.\d+)*)/);
  return m ? m[1]!.split(".").length : null;
}

interface LineRec {
  text: string;
  page: number;
}

/**
 * Extract per-page text from a PDF.
 * Note: pdf-parse returns page numbers as they appear in `pages[].num`.
 */
export async function parsePdf(filePath: string): Promise<ParsedDocument> {
  const data = await readFile(filePath);
  const parser = new PDFParse({ data });
  try {
    const res = await parser.getText();
    const pages: ParsedPage[] = res.pages.map((p) => ({ num: p.num, text: p.text }));
    if (pages.length === 0) {
      throw new Error(`no text extracted — scanned/image-only PDF? (${filePath})`);
    }
    return {
      docName: basename(filePath),
      filePath,
      pageCount: pages.length,
      totalChars: pages.reduce((n, p) => n + p.text.length, 0),
      pages,
    };
  } finally {
    await parser.destroy();
  }
}

/** Chunk a parsed document into structural sections. Deterministic. */
export function chunkDocument(doc: ParsedDocument, opts?: ChunkOptions): DocumentChunk[] {
  const { maxChars, targetChars, overlapFraction, minChars } = { ...DEFAULT_OPTS, ...opts };

  // 1) Flatten to lines with page numbers; keep a heading stack.
  const lines: LineRec[] = [];
  for (const page of doc.pages) {
    for (const raw of page.text.split(/\r?\n/)) {
      const text = raw.trim();
      if (text) lines.push({ text, page: page.num });
    }
  }

  // 2) Group body lines under headings → sections (breadcrumb via heading stack).
  interface Section {
    heading: string | null;
    headingPath: string | null;
    startPage: number | null;
    lines: LineRec[];
  }
  const stack: string[] = [];
  const sections: Section[] = [];
  let current: Section = { heading: null, headingPath: null, startPage: null, lines: [] };

  const flush = () => {
    if (current.lines.length > 0) sections.push(current);
  };

  for (const line of lines) {
    if (isHeading(line.text)) {
      flush();
      const depth = headingDepth(line.text);
      const heading = line.text.replace(/^\s*(?:section\s+)?\d+(?:\.\d+)*[\.\)]?\s*/, "").trim() || line.text.trim();
      if (depth !== null) {
        stack.length = Math.max(0, depth - 1); // pop siblings, keep ancestors
        stack[depth - 1] = heading;
      } else {
        stack.push(heading);
      }
      const path = stack.filter(Boolean).join(" > ");
      current = { heading, headingPath: path, startPage: line.page, lines: [] };
    } else {
      if (current.startPage === null) current.startPage = line.page;
      current.lines.push(line);
    }
  }
  flush();

  // 3) Sections → chunks: split oversize, overlap within a section, merge tiny.
  const chunks: DocumentChunk[] = [];
  let index = 0;

  const pushChunk = (text: string, section: string | null, path: string | null, pageStart: number | null, pageEnd: number | null) => {
    const t = text.trim();
    if (!t) return;
    chunks.push({ chunkIndex: index++, text: t, section, headingPath: path, pageStart, pageEnd });
  };

  for (const sec of sections) {
    const text = sec.lines.map((l) => l.text).join("\n");
    const pages = sec.lines.map((l) => l.page);
    const pStart = pages[0] ?? sec.startPage ?? null;
    const pEnd = pages[pages.length - 1] ?? pStart;

    // Tiny section → merge into previous chunk (keeps context, avoids 2-line chunks)
    if (text.length < minChars && chunks.length > 0) {
      const prev = chunks[chunks.length - 1]!;
      prev.text = `${prev.text}\n\n${text}`.trim();
      prev.pageEnd = pEnd ?? prev.pageEnd;
      continue;
    }

    if (text.length <= maxChars) {
      pushChunk(text, sec.heading, sec.headingPath, pStart, pEnd);
      continue;
    }

    // Oversize section → split on paragraph boundaries with overlap.
    const paragraphs = text.split(/\n{2,}/).map((p) => p.trim()).filter(Boolean);
    let start = 0;
    while (start < paragraphs.length) {
      let end = start;
      let acc = "";
      while (end < paragraphs.length && (acc + paragraphs[end]).length < targetChars) {
        acc += (acc ? "\n\n" : "") + paragraphs[end]!;
        end++;
      }
      if (end === start) {
        // single paragraph bigger than target — hard split by chars
        let remaining = paragraphs[start]!;
        while (remaining.length > maxChars) {
          let cut = remaining.lastIndexOf(" ", maxChars);
          if (cut < targetChars) cut = maxChars;
          pushChunk(remaining.slice(0, cut), sec.heading, sec.headingPath, pStart, pEnd);
          remaining = remaining.slice(cut).trim();
        }
        pushChunk(remaining, sec.heading, sec.headingPath, pStart, pEnd);
        start++;
        continue;
      }
      pushChunk(acc, sec.heading, sec.headingPath, pStart, pEnd);
      // overlap: rewind to include tail of the last paragraph (≈ overlapFraction)
      const overlap = Math.floor(acc.length * overlapFraction);
      if (end < paragraphs.length) {
        const lastPara = paragraphs[end - 1]!;
        const keep = lastPara.slice(-overlap);
        if (keep.length > 30) {
          paragraphs.splice(end, 0, keep);
        }
      }
      start = end;
    }
  }

  return chunks;
}
