#!/usr/bin/env python3
"""Convert suraj520 tickets parquet -> RFC4180 CSV for TS ingest.

Reads config/data/raw/suraj520/tickets.parquet (HF mirror), writes
config/data/tickets/suraj520.csv with pandas to_csv (proper quoting:
commas/quotes/newlines inside fields are escaped; apostrophes preserved).

Usage: python3 scripts/convert-suraj520.py [--parquet PATH] [--out PATH]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["Customer Email", "Product Purchased", "Ticket Type", "Ticket Subject",
           "Combined Text", "Ticket Priority"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(ROOT / "config/data/raw/suraj520/tickets.parquet"))
    ap.add_argument("--out", default=str(ROOT / "config/data/tickets/suraj520.csv"))
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("error: pyarrow not installed (pip install pyarrow)", file=sys.stderr)
        return 1

    table = pq.read_table(args.parquet)
    df = table.to_pandas()
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        print(f"error: parquet missing columns {missing}", file=sys.stderr)
        return 1
    df = df[COLUMNS].copy()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"converted {len(df)} rows x {len(COLUMNS)} cols -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
