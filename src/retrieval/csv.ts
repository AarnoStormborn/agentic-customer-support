/**
 * src/retrieval/csv.ts — minimal RFC 4180 CSV parser (zero deps).
 *
 * Handles the three things naive split(",") breaks on:
 *   - fields wrapped in double quotes (commas inside)
 *   - doubled quotes ("") as an escaped quote inside a quoted field
 *   - literal newlines inside quoted fields (our "Combined Text" is multi-line)
 * Written deliberately instead of pulling csv-parse: suraj520 is 8,469 rows and
 * this is a learner project — ~60 lines beats one more dependency.
 */
export interface CsvRow {
  cells: string[];
  /** 0-based line number in the file where the record started (for error msgs) */
  line: number;
}

export function parseCsv(input: string): CsvRow[] {
  const rows: CsvRow[] = [];
  let cells: string[] = [];
  let cur = "";
  let inQuotes = false;
  let line = 1; // current line while scanning
  let rowStartLine = 1;

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    if (inQuotes) {
      if (ch === '"') {
        if (input[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        if (ch === "\n") line++;
        cur += ch;
      }
    } else if (ch === '"' && cur.length === 0) {
      inQuotes = true;
    } else if (ch === ",") {
      cells.push(cur);
      cur = "";
    } else if (ch === "\n") {
      cells.push(cur);
      rows.push({ cells, line: rowStartLine });
      cells = [];
      cur = "";
      line++;
      rowStartLine = line;
    } else if (ch === "\r") {
      // skip CR; handled by the following \n
    } else {
      cur += ch;
    }
  }
  // trailing record without a final newline
  if (cur.length > 0 || cells.length > 0) {
    cells.push(cur);
    rows.push({ cells, line: rowStartLine });
  }
  return rows;
}
