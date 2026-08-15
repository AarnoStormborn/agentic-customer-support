#!/usr/bin/env bash
# Bulk provision of manuals for vector RAG (Phase 5e — scale up the KB).
# Sources verified by docs/data-research.md §4 (HTTP 200). Idempotent: skips
# files that already exist with a minimum size. Adds appliance manuals from
# archive.org's advancedsearch API so the corpus grows from 3 -> ~30+ manuals.
set -euo pipefail
DATA=config/data/manuals
mkdir -p "$DATA"

fetch() { # $1=url $2=out $3=min_bytes [$4=extra curl args]
  local url="$1" out="$2" min="$3" extra="${4:-}"
  if [[ -f "$out" ]] && [[ $(stat -f%z "$out" 2>/dev/null || stat -c%s "$out") -ge "$min" ]]; then
    echo "skip  $(basename "$out") (exists)"
    return 0
  fi
  echo "fetch $(basename "$out")"
  # shellcheck disable=SC2086
  curl -fL --retry 2 --retry-delay 2 -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" $extra -sS -o "$out" "$url" \
    && [[ $(stat -f%z "$out" 2>/dev/null || stat -c%s "$out") -ge "$min" ]] \
    || { echo "  FAILED $(basename "$out")"; rm -f "$out"; }
}

echo "== Batch 1: verified direct PDFs =="
# LG OLED 55B9 (mirror), HP Pavilion notebook, Dell XPS 13 L321X owner's manual,
# Dell XPS 13 9310 service manual (44MB, needs a UA), Sony Xperia 1 V, Kenmore fridge
fetch "https://media.dustin.eu/media/d200001003283774/oled55b9pla-55-4k-smart-oled-user-manual.pdf" "$DATA/lg_oled_55b9pla.pdf" 500000
fetch "https://media.tatacroma.com/Croma%20Assets/Entertainment/Television/User%20Manual/258426_User%20Manual.pdf" "$DATA/lg_oled_alt_croma.pdf" 400000
fetch "http://www.hp.com/ctg/Manual/bpi04347.pdf" "$DATA/hp_pavilion_notebook_guide.pdf" 400000
fetch "https://dl.dell.com/manuals/all-products/esuprt_laptop/esuprt_xps_laptop/xps-13-l321x_owner%27s%20manual_en-us.pdf" "$DATA/dell_xps13_l321x_owner_manual.pdf" 800000
fetch "https://dl.dell.com/topicspdf/xps-13-9310-laptop_Service-Manual_en-us.pdf" "$DATA/dell_xps13_9310_service_manual.pdf" 10000000
fetch "https://theinformr.com/downloads/cell-phones/manuals/2797/sony-xperia-1-v-manual.pdf" "$DATA/sony_xperia_1v_manual.pdf" 1000000
fetch "https://archive.org/download/Kenmore_25331115308_Refrigerator_User_Manual/Kenmore_25331115308_Refrigerator_User_Manual.pdf" "$DATA/kenmore_fridge_25331115308.pdf" 500000

echo "== Batch 2: archive.org appliance manuals (discovery API) =="
# Find appliance user manuals via the archive.org advancedsearch API, download
# up to N distinct items (size-filtered). Discovery + parse runs in python to
# avoid fragile shell quoting.
LIMIT="${1:-30}"
IDS=$(python3 - "$LIMIT" <<'PY'
import json, sys, urllib.parse, urllib.request
limit = int(sys.argv[1])
q = urllib.parse.quote('title:("user manual" OR "use & care" OR "owner manual") AND (washer OR refrigerator OR microwave OR dishwasher OR range OR dryer) AND mediatype:texts')
url = f"https://archive.org/advancedsearch.php?q={q}&fl[]=identifier&rows=200&output=json"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        d = json.load(r)
    ids = [doc["identifier"] for doc in d.get("response", {}).get("docs", [])]
    # drop known-junk prefixes (dead manual mirrors); keep real-looking items
    bad = ("manualsbase", "manualzilla", "manualsplus", "manua.ls", "manualslib", "manualzz", "media.", "cdl.")
    ids = [i for i in ids if not any(b in i for b in bad)][:limit]
    print("\n".join(ids))
except Exception as e:
    print("", file=sys.stderr)
PY
)
COUNT=0
for id in $IDS; do
  [[ $COUNT -ge $LIMIT ]] && break
  out="$DATA/archive_${id}.pdf"
  if [[ -f "$out" ]]; then
    echo "skip  archive_${id}.pdf (exists)"
    COUNT=$((COUNT+1))
    continue
  fi
  # size-filter: only keep 0.3-15 MB PDFs (skip junk/scans)
  SIZE=$(curl -sSL -A "Mozilla/5.0" -o /dev/null -w "%{size_download}" --max-time 10 "https://archive.org/download/${id}/${id}.pdf" 2>/dev/null || echo 0)
  if [[ "$SIZE" -ge 300000 ]] && [[ "$SIZE" -le 15000000 ]]; then
    echo "fetch archive_${id}.pdf ($SIZE bytes)"
    curl -fL --retry 2 --max-time 90 -A "Mozilla/5.0" -sS -o "$out" "https://archive.org/download/${id}/${id}.pdf" \
      && [[ $(stat -f%z "$out" 2>/dev/null || stat -c%s "$out") -ge 300000 ]] \
      || { echo "  FAILED archive_${id}.pdf"; rm -f "$out"; }
    COUNT=$((COUNT+1))
  else
    echo "skip  ${id} (size $SIZE out of range)"
  fi
done

echo "manuals on disk: $(ls "$DATA"/*.pdf 2>/dev/null | wc -l | tr -d ' ') · size: $(du -sh "$DATA" | cut -f1)"
