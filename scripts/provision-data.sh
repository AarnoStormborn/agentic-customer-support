#!/usr/bin/env bash
# Idempotent data provisioning (docs/design/data-management.md §3.2).
# Re-running skips files that already exist with expected minimum size.
# Also converts the suraj520 parquet -> CSV for TS ingest (needs python3+pyarrow).
set -euo pipefail
DATA=config/data
mkdir -p "$DATA/raw/suraj520" "$DATA/raw/cfpb" "$DATA/raw/comcast" "$DATA/tickets" "$DATA/manuals"

fetch() { # $1=url $2=out $3=min_bytes
  if [[ -f "$2" ]] && [[ $(stat -f%z "$2" 2>/dev/null || stat -c%s "$2") -ge "$3" ]]; then
    echo "skip  $2"; return; fi
  echo "fetch $2"
  curl -fL --retry 3 --retry-delay 2 -C - -o "$2" "$1"
}

# 1) suraj520 tickets (CC0) — HF parquet mirror, no Kaggle auth needed
fetch "https://huggingface.co/datasets/gorkemsevinc/customer_support_tickets/resolve/main/data/train-00000-of-00001.parquet" \
      "$DATA/raw/suraj520/tickets.parquet" 1000000

# 2) CFPB full dump (CC0, ~1.4 GB) + unzip (tickets table at scale, §3.3)
fetch "https://files.consumerfinance.gov/ccdb/complaints.csv.zip" "$DATA/raw/cfpb/complaints.csv.zip" 1000000000
if [[ ! -f "$DATA/raw/cfpb/complaints.csv" ]] || [[ "$DATA/raw/cfpb/complaints.csv.zip" -nt "$DATA/raw/cfpb/complaints.csv" ]]; then
  echo "unzip $DATA/raw/cfpb/complaints.csv.zip"
  unzip -o "$DATA/raw/cfpb/complaints.csv.zip" -d "$DATA/raw/cfpb/"
else
  echo "skip  $DATA/raw/cfpb/complaints.csv (fresh)"
fi

# 2) Manuals (verified URLs, docs/data-research.md §4 / data-management §3.2)
#    NOTE: gscs-b2c.lge.com returned an HTML error page (not a PDF) on 2026-08-11
#    -> using the dustin.eu retail mirror (verified 200).
fetch "https://media.dustin.eu/media/d200001003283774/oled55b9pla-55-4k-smart-oled-user-manual.pdf" \
      "$DATA/manuals/lg_oled_55b9pla.pdf" 500000
fetch "https://theinformr.com/downloads/cell-phones/manuals/2797/sony-xperia-1-v-manual.pdf" \
      "$DATA/manuals/sony_xperia_1v_manual.pdf" 1000000
fetch "https://archive.org/download/Kenmore_25331115308_Refrigerator_User_Manual/Kenmore_25331115308_Refrigerator_User_Manual.pdf" \
      "$DATA/manuals/kenmore_fridge_25331115308.pdf" 500000

# 3) suraj520 parquet -> CSV (RFC4180) for the TS ingest pipeline
if [[ ! -f "$DATA/tickets/suraj520.csv" ]] || [[ "$DATA/raw/suraj520/tickets.parquet" -nt "$DATA/tickets/suraj520.csv" ]]; then
  python3 scripts/convert-suraj520.py
else
  echo "skip  $DATA/tickets/suraj520.csv (fresh)"
fi

echo "provisioning done:"
du -sh "$DATA/raw/suraj520" "$DATA/manuals" "$DATA/tickets" 2>/dev/null || true
