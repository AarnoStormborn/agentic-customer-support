# Data Research

Public datasets for the agentic customer-support retrieval system (SQL + vector RAG + web fallback).
All URLs verified as reachable (HTTP status checked) on 2026-08-11 unless noted. Project schema source: `config/schema.yml`; ingest format source: `config/ingest.py`.

---

## 1. Executive Summary (top recommendation)

**Primary bundle — three datasets that cover all three retrieval modes with permissive licenses:**

| Mode | Dataset | License | Why |
|---|---|---|---|
| SQL + text | **CFPB Consumer Complaint Database** (official, `consumerfinance.gov`) | CC0 (public domain) | ~2.9M complaints, 18 structured fields **plus** free-text consumer narratives. Financial/consumer domain. |
| SQL + text | **Customer Support Ticket Dataset** (Kaggle `suraj520/customer-support-ticket-dataset`, CC0) | CC0 | 8,469 tickets whose product list, ticket types and priorities **match `config/schema.yml` almost exactly** (LG Smart TV, iPhone, Sony Xperia, HP Pavilion, Dell XPS, LG OLED; `technical issue`/`refund request`/`billing inquiry`/…; `critical`/`high`/`medium`/`low`). Drop-in CSV for `ingest_sql_database()`. |
| Vector RAG | **Manufacturer manuals** (LG, Sony, HP, Dell PDFs + archive.org appliance manuals) | Varies (mostly free redistributable; see §4) | Real `.pdf` files for the exact products in the schema. |

**Top fallbacks:** FCC CGB Consumer Complaints (telecom, public domain, structured-only), Comcast telecom complaints CSV (real, 2,224 rows, complaint text + channel), Thinknook "Customer Support on Twitter" (conversations), Twitter US Airline Sentiment (social-media channel + sentiment), Tobi-Bueck customer-support-tickets (61.8k labeled ticket emails, CC BY on Kaggle).

**Recommendation:** Load `tickets` (SQL) from **suraj520 (CC0)** because it is schema-compatible and small (3.95 MB), then enrich the same table or a second table with **CFPB** complaints (narrative → also feeds vector search of complaint text) and **Comcast** tickets (adds telecom + `Received Via` → `ticket_channel`). Point the RAG ingestion at the §4 manual PDFs.

---

## 2. Requirements recap (what the 3 retrieval modes need)

From `config/schema.yml` and `config/ingest.py`:

1. **SQL retrieval** — a CSV with structured ticket fields loaded via `pandas.read_csv → df.to_sql("tickets")`. Expected columns:
   `customer_name, customer_email, customer_age (int), customer_gender, product_purchased (LG Smart TV, iPhone, Sony PlayStation, Google Pixel, HP Pavilion, LG OLED, Dell XPS, Sony Xperia), date_of_purchase (datetime), ticket_type (Refund request, Billing inquiry, Product inquiry, Cancellation request, Technical issue), ticket_priority (Critical, High, Medium, Low), ticket_channel (Social Media, Email, Phone, Chat)`.
2. **Vector retrieval (RAG)** — PDF files in `./config/data/manuals/`, read by `pypdf`, chunked (chunk_size=1024, overlap=50), embedded with `text-embedding-3-small`. Needs technical manual/document text for the products above.
3. **Web search fallback** — no data needed, but datasets with conversation/social text (Twitter-style) best exercise the routing between retrieval modes.

Ideal dataset = structured fields + free text + consumer product/tech-support domain, downloadable (CSV/JSON/Parquet), permissively licensed.

---

## 3. Dataset Catalog

### 3.1 CFPB Consumer Complaint Database — ⭐ primary (SQL + narrative text)

- **Source URL:** https://www.consumerfinance.gov/data-research/consumer-complaints/ (official, US federal gov)
- **Direct downloads (verified HTTP 200):**
  - Full CSV zip: https://files.consumerfinance.gov/ccdb/complaints.csv.zip (1.41 GB)
  - Full JSON zip: https://files.consumerfinance.gov/ccdb/complaints.json.zip (1.57 GB)
  - API docs / field reference: https://cfpb.github.io/api/ccdb/fields.html and https://cfpb.github.io/api/ccdb/api.html
- **Format:** CSV (or JSON) zipped; also official HuggingFace mirror as a loader: https://huggingface.co/datasets/CFPB/consumer-finance-complaints (`cc0-1.0`, 1M+ rows, loads via `datasets` library; not parquet, executes loader script).
- **Size:** ~2.9M complaints; CSV zip 1.41 GB; JSON zip 1.57 GB. Updates daily.
- **License:** CC0 — public domain (US federal government work).
- **Structured fields (18):** `Date received, Product, Sub-product, Issue, Sub-issue, Company, State, ZIP code, Submitted via, Date sent to company, Company response to consumer, Timely response?, Consumer disputed?, Complaint ID, Company public response, Tags, Consumer consent provided?, **Consumer complaint narrative** (free text)`.
- **Text fields:** `Consumer complaint narrative` (consumer-written story, only for complaints where the consumer opted in, PII scrubbed) and `Company public response` (canned response text).
- **Mapping:**
  - SQL: `Product/Sub-product` ↔ `product_purchased`; `Issue` ↔ `ticket_type`; `Date received` ↔ `date_of_purchase`; `Submitted via` (Web/Phone/Postal mail/Referral/Email/Fax) ↔ `ticket_channel`; `State/ZIP` + `Company` = extra structured columns (needs schema extension).
  - Vector: embed the narrative text (complaint stories are exactly the "customer says it in their own words" retrieval case).
  - Web: narrative covers products/companies outside the manual set (e.g., debt collection, mortgages) — good fallback-to-web demo.
- **Ingestion notes:** 1.4 GB is heavy for a demo. Recommended: download the **filtered export** from the search UI (filter `has narrative = yes`, or a single product, then "Export data"), or use the HF loader. Filtered CSV exports are a few MB–tens of MB. Map columns with a small `rename` step before `to_sql`.
- **Caveats:** Not a statistical sample (self-selected complainants); narratives are a minority of records and arrive with a lag; PII already scrubbed by CFPB.

### 3.2 Customer Support Ticket Dataset (Kaggle `suraj520`) — ⭐ primary (schema-compatible SQL)

- **Source URL:** https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
- **Direct download:** `kaggle datasets download -d suraj520/customer-support-ticket-dataset` (file `customer_support_tickets.csv`, 3.95 MB); mirror on HuggingFace (parquet, 1.12 MB): https://huggingface.co/datasets/gorkemsevinc/customer_support_tickets — parquet: `https://huggingface.co/datasets/gorkemsevinc/customer_support_tickets/resolve/main/data/train-00000-of-00001.parquet`
- **Format:** CSV / Parquet; 8,469 rows, 6 columns (verified by downloading the parquet).
- **License:** CC0 — Public Domain (verified on the Kaggle page).
- **Structured fields:** `Customer Email, Product Purchased, Ticket Type, Ticket Priority` (+ `Ticket Subject`).
- **Text fields:** `Combined Text` (multi-sentence ticket narrative), `Ticket Subject`.
- **Verified content (sampled from parquet):**
  - `Product Purchased` values: `lg smart tv, lg oled, iphone, sony xperia, hp pavilion, dell xps, sony 4k hdr tv, canon eos, gopro hero, nest thermostat, philips hue, amazon echo, roomba robot vacuum, apple airpods, microsoft surface, dyson vacuum, google nest, …`
  - `Ticket Type` values: `refund request, technical issue, cancellation request, product inquiry, billing inquiry` — **exact match to `schema.yml`**.
  - `Ticket Priority` values: `critical, high, medium, low` — **exact match to `schema.yml`**.
- **Mapping:**
  - SQL: direct 1:1 for `customer_email, product_purchased, ticket_type, ticket_priority`. Missing from schema: `customer_name, customer_age, customer_gender, date_of_purchase, ticket_channel` (synthesize deterministically or extend schema; see §5).
  - Vector: `Combined Text` is ideal chunk source; subject line also usable.
- **Ingestion notes:** drop-in CSV for `ingest_sql_database()`. Add a small script to derive `customer_name` from email, draw `customer_age`/`gender`/`date_of_purchase`/`ticket_channel` randomly (seed fixed) to complete the schema.
- **Caveats:** **Synthetic** (template-generated text; many rows end with "resolution pending"; repetitive phrasing). Fine for pipeline/agent demo purposes, not for training models. No purchase dates/channels originally.

### 3.3 Customer Support on Twitter (Thinknook / Stuart Axelbrooke) — conversation text + social channel

- **Source URL:** https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- **Format:** CSV; ~3.98M rows total across `tweets.csv` + `responses.csv` (~516 MB combined; compiled 2017 by Stuart Axelbrooke, Thinknook).
- **License:** No explicit license on Kaggle page (research use; Twitter ToS applies — see §6). HF mirror (text-only): https://huggingface.co/datasets/gorkemsevinc/Customer_Support_on_Twitter (3,982,937 rows, 195 MB, **single column `cleaned_text`** — degraded; prefer Kaggle original).
- **Structured fields (Kaggle original):** `tweet_id, author_id (anonymized), inbound (bool), created_at, response_tweet_id, in_response_to_tweet_id` (thread reconstruction).
- **Text fields:** `text` (tweet), response text (2nd CSV).
- **Brands covered:** Apple, Amazon, Uber, Delta, Spotify and other large consumer brands.
- **Mapping:**
  - SQL: `created_at` ↔ `date_of_purchase`-style timestamp; `inbound` ↔ direction; author_id ↔ customer_id; brand ↔ product/company dimension.
  - Vector: tweet + response pairs = support-conversation corpus (separate from manuals).
  - Web: original tweets are public social posts — the natural "what do customers say about X" fallback source.
- **Ingestion notes:** download via Kaggle CLI; reconstruct conversations by joining `tweets` and `responses` on `tweet_id`. Consider a subsample (e.g., one brand) to keep Postgres/pgvector load low.
- **Caveats:** License unclear (research-only in practice; commercial use risky); Twitter/X ToS on redistribution; HF mirror is text-only (no structured fields).

### 3.4 Twitter US Airline Sentiment (CrowdFlower/Figure Eight) — social channel + sentiment

- **Source URL:** https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment ; HF: https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment
- **Format:** CSV `Tweets.csv` (14,640 tweets, Feb 2015) + `database.sqlite` on HF.
- **License:** **CC BY-NC-SA 4.0** (non-commercial, share-alike) — demo/edu use only.
- **Structured fields:** `airline, airline_sentiment, airline_sentiment_confidence, negativereason, negativereason_confidence, retweet_count, tweet_created, tweet_location, user_timezone, name`.
- **Text fields:** `text`.
- **Mapping:** SQL: `airline` ↔ brand dimension, `negativereason` ↔ issue category, sentiment ↔ outcome; Vector: tweet text; Web: social-media support complaints.
- **Caveats:** 2015 data; NC license.

### 3.5 FCC CGB Consumer Complaints (telecom) — structured-only fallback

- **Source URL:** https://opendata.fcc.gov/Consumer/CGB-Consumer-Complaints-Data/3xyp-aqkj (also https://www.fcc.gov/consumer-help-center-data)
- **Direct download (verified 200):** `https://opendata.fcc.gov/api/views/3xyp-aqkj/rows.csv?accessType=DOWNLOAD` (Socrata API; CSV generated on demand; also JSON via `rows.json`)
- **Format:** CSV/JSON via Socrata API; informal consumer complaints since Oct 31, 2014 (hundreds of thousands of records; API supports filters/pagination).
- **License:** US government public data (public domain).
- **Structured fields (verified header):** `Ticket ID, Ticket Created, Date Created, Date of Issue, Time of Issue, Form (TV/Phone/Internet/Broadcast…), Method (Wireless/Wired/VoIP…), Issue, Caller ID Number, Type of Call or Message, Advertiser Business Number, City, State, Zip, Location, Type of Property Goods or Services`.
- **Text fields:** **none** (no complaint narrative is published; "Issue" is a categorical label). This is the key caveat vs. CFPB.
- **Mapping:** SQL: `Form/Method/Issue` ↔ ticket type/category; `Ticket Created` ↔ date; `State/City/Zip` ↔ geography. Good telecom structured table; text must come from another source.
- **Ingestion notes:** use Socrata API with `$where`/`$limit` filters to pull a manageable subset (e.g., one year or one Form).

### 3.6 Comcast Telecom Consumer Complaints CSV (GitHub) — real telecom text + channel

- **Source URL:** https://github.com/kuhimans/Comcast-Telecom-Consumer-Complaints-Analysis
- **Direct download (verified 200):** `https://raw.githubusercontent.com/kuhimans/Comcast-Telecom-Consumer-Complaints-Analysis/master/Comcast_telecom_complaints_data.csv` (2,224 rows incl. header; ~1 MB)
- **License:** No license file in repo (data derived from FCC complaints; treat as public-domain-derived, verify before commercial redistribution).
- **Structured fields (verified header):** `Ticket #, Date, Date_month_year, Time, Received Via, City, State, Zip code, Status, Filing on Behalf of Someone`.
- **Text fields:** `Customer Complaint` (short free-text like "Comcast Cable Internet Speeds", "Speed and Service").
- **Mapping:** SQL: `Ticket #` ↔ id; `Date` ↔ `date_of_purchase`; **`Received Via` ↔ `ticket_channel`** (values: Customer Care Call, Internet, …); `Status` ↔ resolution state. Vector: complaint text. Great small second table for telecom.
- **Caveats:** Small, 2015-era, short complaint text; repo unmaintained.

### 3.7 Tobi-Bueck "Customer Support Tickets" (HF) / "Customer IT Support – Ticket Dataset" (Kaggle) — labeled ticket emails

- **Source URLs:** HF: https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets (61,800 rows, CSV, **CC BY-NC 4.0**); Kaggle original: https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets (**CC BY 4.0** — the attribution version)
- **Files (HF, verified):** `aa_dataset-tickets-multi-lang-5-2-50-version.csv`, `dataset-tickets-german_normalized_50_5_2.csv`, `dataset-tickets-multi-lang-4-20k.csv` (~69 MB total).
- **Structured fields:** `type` (category/queue: technical support, customer service, billing & payments, product support, IT support…), priority, language, (queue metadata).
- **Text fields:** `subject` (customer email subject), `body` (customer email), `answer` (agent's first response).
- **Mapping:** SQL: `type` ↔ `ticket_type`, priority ↔ `ticket_priority`; Vector: `body`/`answer` = question-answer pairs for RAG; Web: mixed.
- **Caveats:** Synthetic-ish IT/helpdesk emails; the HF mirror is **non-commercial** (CC BY-NC) — use the Kaggle CC BY 4.0 version for commercial use.

### 3.8 IT Support Tickets — synthetic SaaS helpdesk (Kaggle `ahsanneural`) — fallback

- **Source URL:** https://www.kaggle.com/datasets/ahsanneural/synthetic-it-support-tickets
- **Format:** `synthetic_it_support_tickets.csv` — 100,000 rows × 20 columns.
- **License:** no explicit license on the page (synthetic, "freely usable for experimentation"; verify before commercial use).
- **Fields:** ticket id/date/type/priority/category/assignment + rich description text, SLA, sentiment, CSAT.
- **Mapping:** SQL: rich structured set (priority, category, SLA, sentiment, CSAT); Vector: descriptions. Good scale for load testing.
- **Caveats:** synthetic; IT-support domain (not consumer electronics); license not explicit.

### 3.9 Other verified options (secondary)

- **CFPB mirrors on HuggingFace (parquet, CFPB-derived):**
  - `determined-ai/consumer_complaints_medium` (13.5 MB parquet, train/test; license not declared — CFPB source is CC0, but confirm before shipping) — https://huggingface.co/datasets/determined-ai/consumer_complaints_medium
  - `AdiOO7/Bank_Complaints` (Apache 2.0, `complaints.json`) — https://huggingface.co/datasets/AdiOO7/Bank_Complaints
- **Console-AI/IT-helpdesk-synthetic-tickets** (HF, **MIT**, `tickets.csv`): https://huggingface.co/datasets/Console-AI/IT-helpdesk-synthetic-tickets
- **h3en1x/audio-retailer-customer-support-tickets** (HF, synthetic consumer-electronics retail tickets, ~198k rows, no license): https://huggingface.co/datasets/h3en1x/audio-retailer-customer-support-tickets
- **HashiruGunathilake/support-tickets-telecommunication** (HF, 27,229 rows, no license): https://huggingface.co/datasets/HashiruGunathilake/support-tickets-telecommunication
- **Bank and Credit Card Complaints (Kaggle `mexwell`):** CFPB subset with narrative — https://www.kaggle.com/datasets/mexwell/bank-and-credit-card-complaints (verify license on page before use).
- **Zendesk-branded public dumps:** effectively **not available** — no canonical real Zendesk ticket dump was found; only exporters exist (e.g. https://github.com/klausbadelt/zendesk-ticket-export exports from a live account). If a Zendesk look-and-feel is required, the Tobi-Bueck and suraj520 datasets are the practical stand-ins.

---

## 4. Manuals / Docs sources (for vector RAG)

All URLs below verified reachable (HTTP 200) on 2026-08-11. Place PDFs in `./config/data/manuals/` — `RAGIngestion.upsert_docs()` picks up every `*.pdf` in that directory.

### 4.1 Direct PDFs (verified)

| Product (schema) | Source | Verified size |
|---|---|---|
| LG Smart TV / LG OLED | LG OLED55B9PLA Owner's Manual (retail CDN mirror): https://media.dustin.eu/media/d200001003283774/oled55b9pla-55-4k-smart-oled-user-manual.pdf | 770 KB ✓ |
| LG OLED (alt) | https://media.tatacroma.com/Croma%20Assets/Entertainment/Television/User%20Manual/258426_User%20Manual.pdf | 726 KB ✓ |
| LG TV (any model, official) | https://www.lg.com/us/support/manuals-documents (LG blocks plain curl — use browser or a mirror) | — |
| Sony Xperia | Xperia 1 V manual (theinformr CDN): https://theinformr.com/downloads/cell-phones/manuals/2797/sony-xperia-1-v-manual.pdf | 2.2 MB ✓ |
| Sony Xperia (official page) | https://www.sony.com/electronics/support/mobile-phones-tablets-mobile-phones/xperia-1-v-256gb/manuals | — |
| HP Pavilion | HP Pavilion Notebook PC User's Guide (hp.com): http://www.hp.com/ctg/Manual/bpi04347.pdf | 758 KB ✓ |
| HP Pavilion (modern) | https://support.hp.com/us-en/product/hp-pavilion-15-cc000-laptop-pc/15551391/manuals | — |
| Dell XPS | XPS 13 9310 Service Manual (dl.dell.com): https://dl.dell.com/topicspdf/xps-13-9310-laptop_Service-Manual_en-us.pdf (send a browser User-Agent) | 44.7 MB ✓ |
| Dell XPS (alt) | XPS 13 (L321X) Owner's Manual: `https://dl.dell.com/manuals/all-products/esuprt_laptop/esuprt_xps_laptop/xps-13-l321x_owner's%20manual_en-us.pdf` | 2.8 MB ✓ |
| Google Pixel | **No official big user-guide PDF** — Google ships Pixel with an online Help Center. Options: ManualsLib Pixel 7 manual https://www.manualslib.com/manual/2876995/Google-Pixel-7.html; Google Pixel Help Center HTML (crawl `support.google.com/pixelphone/`), safety/regulatory pages https://support.google.com/product-documentation/answer/9204905; devicebeast/manuals.plus user-guide PDFs (third-party). | — |
| iPhone (bonus) | Apple support manuals/manuals at https://support.apple.com/manuals (HTML/PDF per device) | — |

### 4.2 Home-appliance manuals (archive.org — verified downloadable, public archive)

Advanced-search hit count: **5,410 appliance-manual items** on archive.org. Verified examples (download URL pattern: `https://archive.org/download/<identifier>/<identifier>.pdf`):
- Kenmore Refrigerator 25331115308 — `https://archive.org/download/Kenmore_25331115308_Refrigerator_User_Manual/Kenmore_25331115308_Refrigerator_User_Manual.pdf` ✓ (302 → CDN mirror, downloads fine)
- Panasonic NN-SN747 Microwave Oven — `https://archive.org/download/Panasonic_NN-SN747_Microwave_Oven_User_Manual/Panasonic_NN-SN747_Microwave_Oven_User_Manual.pdf` ✓
- Amana Bottom-Freezer Refrigerator ARS266KBW (`manualsbase-id-574114`), Frigidaire FRS26R2AQD (`manualsbase-id-564792`), Kenmore Elite 721.88512 microwave, Kenmore TRIO 795.7756 fridge — all found via `https://archive.org/advancedsearch.php?q="user manual" AND (washing machine OR refrigerator OR microwave)`
- Bulk discovery API: `https://archive.org/advancedsearch.php?q=...&fl[]=identifier&rows=100&output=json` then download via `https://archive.org/download/<id>/<id>.pdf`

### 4.3 Online manual aggregators (per-manual PDF download, good for a few more products)
- **ManualsLib** — https://www.manualslib.com/ (LG OLED `manual/2052512/Lg-Oled-Tv.html`, HP Pavilion `manual/711736/Hp-Pavilion.html`, Google Pixel 7 `manual/2876995/Google-Pixel-7.html`; verified 200)
- **manua.ls** — https://www.manua.ls/televisions/lg ; **manuals.plus** — https://manuals.plus/ ; **DeviceBeast** — https://devicebeast.com/phone-user-manual/pixel-9-pro
- **LG eGuide online manual** (HTML, crawlable): https://eguide.lgappstv.com/manual/gb/index.html

---

## 5. Recommended dataset bundle

### Final pick
1. **SQL `tickets` table ← `suraj520/customer-support-ticket-dataset`** (CC0, 8,469 rows, schema-compatible products/types/priorities). Optionally augment columns to match `schema.yml` exactly (derive `customer_name` from email; synthesize `customer_age`, `customer_gender`, `date_of_purchase`, `ticket_channel` with a fixed-seed RNG in a preprocessing script).
2. **SQL second table (or same) ← CFPB complaints** (CC0) — richest real structured+narrative data; filter to a subset before download (see commands).
3. **Vector RAG manuals ← §4 PDF set** (LG OLED, HP Pavilion, Dell XPS, Sony Xperia, Google Pixel via ManualsLib/Help Center, + 2–3 archive.org appliance manuals). ~6–10 PDFs, 1–60 MB each.

### Fallbacks (in order)
- **FCC CGB complaints** (public domain, telecom, structured-only) — if a bigger SQL corpus is needed; pair with Comcast CSV for text.
- **Comcast telecom complaints CSV** (real text + `Received Via` channel) — small, quick win for telecom coverage.
- **Thinknook Customer Support on Twitter** — conversation text for vector/web modes (license caveat; subsample).
- **Twitter US Airline Sentiment** — social-media channel demo (NC license).
- **Tobi-Bueck tickets (Kaggle CC BY 4.0)** — 61.8k labeled email tickets with agent answers.

### Download commands
```bash
mkdir -p config/data/tickets config/data/manuals

# 1) suraj520 tickets — CC0, schema-compatible (via HF parquet mirror, no Kaggle auth)
curl -L -o config/data/tickets/tickets.parquet \
  https://huggingface.co/datasets/gorkemsevinc/customer_support_tickets/resolve/main/data/train-00000-of-00001.parquet
# (or with Kaggle CLI:  kaggle datasets download -d suraj520/customer-support-ticket-dataset -p config/data/tickets)
# convert to CSV:  python -c "import pandas as pd; pd.read_parquet('config/data/tickets/tickets.parquet').to_csv('config/data/tickets/tickets.csv', index=False)"

# 2) CFPB — full dump (1.41 GB; for a lighter start use the search UI with filters, or the HF loader)
curl -L -o config/data/tickets/complaints.csv.zip https://files.consumerfinance.gov/ccdb/complaints.csv.zip
unzip -o config/data/tickets/complaints.csv.zip -d config/data/tickets/
# Filtered alternative (has narrative only, ~smaller): use the CFPB search UI → Export
#   https://www.consumerfinance.gov/data-research/consumer-complaints/search/?has_narrative=true

# 3) Comcast telecom complaints — real text + channel
curl -L -o config/data/tickets/comcast_complaints.csv \
  https://raw.githubusercontent.com/kuhimans/Comcast-Telecom-Consumer-Complaints-Analysis/master/Comcast_telecom_complaints_data.csv

# 4) Manuals (RAG) — examples
curl -L -o "config/data/manuals/lg_oled_55b9pla.pdf" https://media.dustin.eu/media/d200001003283774/oled55b9pla-55-4k-smart-oled-user-manual.pdf
curl -L -o "config/data/manuals/hp_pavilion_user_guide.pdf" http://www.hp.com/ctg/Manual/bpi04347.pdf
curl -L -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126.0" \
  -o "config/data/manuals/dell_xps13_9310_service_manual.pdf" \
  https://dl.dell.com/topicspdf/xps-13-9310-laptop_Service-Manual_en-us.pdf
curl -L -o "config/data/manuals/sony_xperia_1v_manual.pdf" \
  https://theinformr.com/downloads/cell-phones/manuals/2797/sony-xperia-1-v-manual.pdf
curl -L -o "config/data/manuals/kenmore_fridge_25331115308.pdf" \
  https://archive.org/download/Kenmore_25331115308_Refrigerator_User_Manual/Kenmore_25331115308_Refrigerator_User_Manual.pdf

# 5) Conversation datasets (fallbacks)
# kaggle datasets download -d thoughtvector/customer-support-on-twitter -p config/data/tickets
# kaggle datasets download -d crowdflower/twitter-airline-sentiment -p config/data/tickets
# kaggle datasets download -d tobiasbueck/multilingual-customer-support-tickets -p config/data/tickets   # CC BY 4.0
```

Then run `python config/ingest.py` (with `DB_STRING` set) — it loads `config/data/tickets/tickets.csv` into `tickets` and ingests `config/data/manuals/*.pdf` into `t_docs`/`t_docs_chunks`.

---

## 6. Risk & license notes

- **CFPB (CC0, public domain)** — safest dataset here; federal government work, no restrictions, no PII (Bureau scrubs).
- **FCC (public domain)** — same, but structured-only (no narrative); Socrata API rate limits apply to big pulls.
- **suraj520 (CC0)** — clean license; synthetic data (templates, repeated phrasing, "resolution pending"). OK for pipeline demos, misleading as a model-training corpus.
- **Thinknook Customer Support on Twitter** — **no explicit license**; redistribution conflicts with Twitter/X Terms of Service; treat as research-only. The HF mirror strips all structured columns (only `cleaned_text`) — don't use it for the SQL mode.
- **Twitter US Airline Sentiment — CC BY-NC-SA 4.0** — non-commercial; attribution + share-alike required. Fine for this project (research/learning), not for a commercial product.
- **Tobi-Bueck**: Kaggle copy is **CC BY 4.0** (attribution, commercial OK); HF mirror is **CC BY-NC 4.0** (non-commercial) — prefer the Kaggle file if licensing matters.
- **Comcast CSV (GitHub)** — repo has no license; data is FCC-derived (public domain provenance), but the specific compiled file has unclear terms — keep for internal use.
- **ahsanneural, h3en1x, HashiruGunathilake, determined-ai** — no explicit license on most; **Console-AI is MIT** (safe). Verify licenses before commercial redistribution.
- **Manual PDFs** — manufacturer manuals (LG, HP, Dell, Sony, Google) are copyright of the vendors but freely distributed for customer use; internal RAG ingestion is fine, do not republish. archive.org manuals are preserved public-archive items — check the item page for any access restrictions (most are open).
- **Schema mismatch to plan for** — no single dataset provides every column in `schema.yml` (age/gender/name/purchase-date/channel are spread across datasets or absent). Plan a small normalization script; the SQL agent should also tolerate extra/missing columns (e.g., CFPB has `state/zip/company`, suraj520 lacks `date_of_purchase`).
- **Sizes** — CFPB full dump is 1.4–1.6 GB; prefer filtered exports or the HF loader for the demo; Thinknook ~516 MB; everything else ≤ 70 MB.

*Research performed 2026-08-11; all URLs verified via HTTP HEAD/GET. Kaggle pages require a (free) account for CLI downloads.*
