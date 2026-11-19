"""Retrieve and filter GenBank records for the APBT study cohort."""

import os, sys, re, time, csv, random, argparse
from datetime import datetime

try:
    from Bio import Entrez, SeqIO
except ImportError:
    print("\nERROR: Biopython not installed.  Run:  pip install biopython\n")
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

Entrez.email = os.environ.get("NCBI_EMAIL")
Entrez.tool  = "APBT_Study3_SequenceRetrieval_v3"

DEFAULT_SAVE_DIR = "data"

# Quality thresholds applied during retrieval
MIN_SEQ_LENGTH  = 28000    # minimum genome length in nucleotides
MAX_SEQ_LENGTH  = 31000    # maximum (avoids concatenated records)
N_MAX_PCT       = 1.0      # maximum % ambiguous (N) bases

# Whether to accept direct child sublineages of the search target
ALLOW_SUBLINEAGES = True

# Target lineages — 8 major Pango groups
LINEAGE_TARGETS = [
    "B.1.1.7",    # Alpha        — 2020–21 VOC
    "P.1",        # Gamma        — 2021 VOC
    "BQ.1",       # Omicron BQ.1 — late 2022
    "XBB.1.5",    # Omicron XBB.1.5 — 2022-23 XBB recombinant
    "B.1.617.2",  # Delta        — mid-2021 VOC
    "BA.1",       # Omicron BA.1 — late 2021
    "BA.2",       # Omicron BA.2 — early 2022
    "BA.5",       # Omicron BA.5 — mid-2022
]

WHO_LABELS = {
    "B.1.1.7":"Alpha",       "P.1":"Gamma",
    "BQ.1":"Omicron BQ.1",   "XBB.1.5":"Omicron XBB.1.5 (Kraken)",
    "B.1.617.2":"Delta",     "BA.1":"Omicron BA.1",
    "BA.2":"Omicron BA.2",   "BA.5":"Omicron BA.5",
}

# CSV output columns
CSV_COLUMNS = [
    "accession", "lineage", "detected_lineage", "lineage_target",
    "collection_date", "country", "sequence_length", "gc_content",
    "n_content_pct", "description", "sequence"
]


# ═════════════════════════════════════════════════════════════════════════════
# LINEAGE MATCHING
# ═════════════════════════════════════════════════════════════════════════════

def is_acceptable_lineage(detected, target):
    """Return True if detected lineage is acceptable for target search."""
    if detected is None:
        return False
    if detected == target:
        return True
    if ALLOW_SUBLINEAGES:
        prefix = target + "."
        if detected.startswith(prefix):
            return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# SEQUENCE QUALITY CHECKS
# ═════════════════════════════════════════════════════════════════════════════

def passes_quality(seq):
    """Return (passes: bool, reason: str)."""
    length = len(seq)
    if length < MIN_SEQ_LENGTH:
        return False, f"too short ({length} nt)"
    if length > MAX_SEQ_LENGTH:
        return False, f"too long ({length} nt)"
    n_count = seq.upper().count("N")
    n_pct   = n_count / length * 100
    if n_pct > N_MAX_PCT:
        return False, f"N-content {n_pct:.1f}% > {N_MAX_PCT}%"
    return True, "ok"


def gc_content(seq):
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return round(gc / len(seq) * 100, 2) if seq else 0.0


def n_content(seq):
    return round(seq.upper().count("N") / len(seq) * 100, 3) if seq else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# LINEAGE EXTRACTION FROM GENBANK RECORD
# ═════════════════════════════════════════════════════════════════════════════

PANGO_RE = re.compile(
    r'(?<![A-Za-z0-9])'
    r'([A-Z]{1,2}\.[0-9]+(?:\.[0-9]+)*'
    r'|[A-Z]{2,3}[0-9]*\.[0-9]+(?:\.[0-9]+)*)'
    r'(?![A-Za-z0-9])'
)

def extract_lineage(record):
    """Extract Pango lineage from a GenBank record."""
    for feat in record.features:
        for qualifier in ["note","isolate","strain","db_xref",
                          "collected_by","isolation_source"]:
            for val in feat.qualifiers.get(qualifier, []):
                m = PANGO_RE.search(str(val))
                if m:
                    candidate = m.group(1)
                    if not re.match(r'^[A-Z]{2}[0-9]', candidate):
                        return candidate

    for key in ["comment","data_file_division","keywords"]:
        val = str(record.annotations.get(key, ""))
        m = PANGO_RE.search(val)
        if m:
            candidate = m.group(1)
            if not re.match(r'^[A-Z]{2}[0-9]', candidate):
                return candidate

    m = PANGO_RE.search(record.description)
    if m:
        candidate = m.group(1)
        if not re.match(r'^[A-Z]{2}[0-9]', candidate):
            return candidate

    return None


# ═════════════════════════════════════════════════════════════════════════════
# COUNTRY EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

ISO3_TO_COUNTRY = {
    "USA":"USA","GBR":"United Kingdom","DEU":"Germany","FRA":"France",
    "NLD":"Netherlands","JPN":"Japan","IND":"India","BRA":"Brazil",
    "ZAF":"South Africa","DNK":"Denmark","ESP":"Spain","CAN":"Canada",
    "AUS":"Australia","ITA":"Italy","BEL":"Belgium","SWE":"Sweden",
    "POL":"Poland","PRT":"Portugal","CHE":"Switzerland","MEX":"Mexico",
    "THA":"Thailand","CHN":"China","KOR":"South Korea","ISR":"Israel",
    "TUR":"Turkey","GRC":"Greece","AUT":"Austria","CZE":"Czech Republic",
    "NOR":"Norway","FIN":"Finland","HUN":"Hungary","ROU":"Romania",
    "SVK":"Slovakia","SVN":"Slovenia","HRV":"Croatia","BGR":"Bulgaria",
    "SRB":"Serbia","UKR":"Ukraine","RUS":"Russia","KAZ":"Kazakhstan",
    "SGP":"Singapore","MYS":"Malaysia","IDN":"Indonesia","PHL":"Philippines",
    "VNM":"Vietnam","PAK":"Pakistan","BGD":"Bangladesh","LKA":"Sri Lanka",
    "EGY":"Egypt","NGA":"Nigeria","KEN":"Kenya","GHA":"Ghana",
    "ETH":"Ethiopia","COD":"DR Congo","ZAR":"DR Congo","TZA":"Tanzania",
    "UGA":"Uganda","SEN":"Senegal","CIV":"Ivory Coast","CMR":"Cameroon",
    "ARG":"Argentina","CHL":"Chile","COL":"Colombia","PER":"Peru",
    "VEN":"Venezuela","ECU":"Ecuador","CRI":"Costa Rica","PAN":"Panama",
    "NZL":"New Zealand","FJI":"Fiji","PNG":"Papua New Guinea",
}

def extract_country(record):
    """Extract country from GenBank record."""
    for feat in record.features:
        if feat.type != "source":
            continue
        for qname in ["geo_loc_name", "country"]:
            vals = feat.qualifiers.get(qname, [])
            if vals:
                raw = vals[0].split(":")[0].strip()
                if raw and raw.lower() not in ("unknown","not provided","n/a","na"):
                    return raw

    m = re.search(r'/human/([A-Z]{2,3})/', record.description)
    if m:
        code = m.group(1)
        if code in ISO3_TO_COUNTRY:
            return ISO3_TO_COUNTRY[code]
        two_letter = {
            "US":"USA","GB":"United Kingdom","DE":"Germany","FR":"France",
            "NL":"Netherlands","JP":"Japan","IN":"India","BR":"Brazil",
            "ZA":"South Africa","DK":"Denmark","ES":"Spain","CA":"Canada",
            "AU":"Australia","IT":"Italy","BE":"Belgium","SE":"Sweden",
        }
        if code in two_letter:
            return two_letter[code]
    return ""


def extract_date(record):
    """Extract collection date from source feature."""
    for feat in record.features:
        if feat.type == "source":
            for qname in ["collection_date","collection date"]:
                vals = feat.qualifiers.get(qname, [])
                if vals:
                    return vals[0]
    return record.annotations.get("date","")


# ═════════════════════════════════════════════════════════════════════════════
# NCBI SEARCH AND FETCH
# ═════════════════════════════════════════════════════════════════════════════

def build_query(lineage):
    """Build NCBI search query for a specific lineage."""
    return (
        f'"{lineage}"[All Fields] '
        f'AND "Severe acute respiratory syndrome coronavirus 2"[Organism] '
        f'AND "complete genome"[Title] '
        f'AND {MIN_SEQ_LENGTH}:{MAX_SEQ_LENGTH}[Sequence Length]'
    )


def search_ncbi(lineage, n_target):
    """Search NCBI. Returns (ids, web_env, query_key, total)."""
    query = build_query(lineage)
    print(f"    Query: {query[:90]}...")
    try:
        handle = Entrez.esearch(
            db="nuccore", term=query,
            retmax=n_target * 5 if n_target > 0 else 10000,
            usehistory="y")
        result = Entrez.read(handle)
        handle.close()
        total = int(result["Count"])
        print(f"    NCBI total results: {total:,}")
        return result["IdList"], result["WebEnv"], result["QueryKey"], total
    except Exception as e:
        print(f"    ERROR searching NCBI: {e}")
        return [], None, None, 0


def fetch_batch(web_env, query_key, start, batch_size):
    """Fetch one batch of GenBank records."""
    try:
        handle = Entrez.efetch(
            db="nuccore", rettype="gb", retmode="text",
            retstart=start, retmax=batch_size,
            webenv=web_env, query_key=query_key)
        records = list(SeqIO.parse(handle, "genbank"))
        handle.close()
        return records
    except Exception as e:
        print(f"      Batch fetch error: {e} — retrying in 5s...")
        time.sleep(5)
        try:
            handle = Entrez.efetch(
                db="nuccore", rettype="gb", retmode="text",
                retstart=start, retmax=batch_size,
                webenv=web_env, query_key=query_key)
            records = list(SeqIO.parse(handle, "genbank"))
            handle.close()
            return records
        except Exception as e2:
            print(f"      Retry failed: {e2}")
            return []


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RETRIEVAL LOOP
# ═════════════════════════════════════════════════════════════════════════════

def retrieve_lineage(lineage, n_target, seen_accessions, seen_sequences, batch_size=50):
    """
    Retrieve up to n_target clean, unique sequences for one lineage.
    If n_target=0, retrieve ALL available sequences.
    """
    print(f"\n  [{LINEAGE_TARGETS.index(lineage)+1}/{len(LINEAGE_TARGETS)}]  "
          f"{lineage}  ({WHO_LABELS.get(lineage,'')})")

    ids, web_env, query_key, total = search_ncbi(lineage, n_target)
    if not ids:
        return [], {"lineage":lineage,"searched":0,"accepted":0,
                    "rej_dup_acc":0,"rej_dup_seq":0,
                    "rej_quality":0,"status":"NO_RESULTS"}

    # Determine how many to fetch
    max_to_fetch = n_target if n_target > 0 else total
    fetch_limit = min(len(ids), max_to_fetch * 3 if n_target > 0 else 10000)
    ids = ids[:fetch_limit]
    print(f"    Fetching up to: {len(ids)}")

    records_out = []
    stats = {"lineage":lineage,"searched":total,"accepted":0,
             "rej_dup_acc":0,"rej_dup_seq":0,
             "rej_quality":0,"status":"OK"}

    start = 0
    while len(records_out) < max_to_fetch and start < len(ids):
        end = min(start + batch_size, len(ids))
        print(f"    Batch {start+1}–{end} ...", end=" ", flush=True)

        batch = fetch_batch(web_env, query_key, start, batch_size)
        accepted_this_batch = 0

        for rec in batch:
            if n_target > 0 and len(records_out) >= n_target:
                break

            accession = rec.id
            seq       = str(rec.seq).upper()
            seq_clean = re.sub(r"[^ACGTN]", "N", seq)

            if accession in seen_accessions:
                stats["rej_dup_acc"] += 1
                continue

            if seq_clean in seen_sequences:
                stats["rej_dup_seq"] += 1
                continue

            ok, reason = passes_quality(seq_clean)
            if not ok:
                stats["rej_quality"] += 1
                continue

            detected = extract_lineage(rec)
            country = extract_country(rec)
            date    = extract_date(rec)

            seen_accessions.add(accession)
            seen_sequences.add(seq_clean)

            records_out.append({
                "accession":       accession,
                "lineage":         lineage,
                "detected_lineage": detected,
                "lineage_target":  lineage,
                "collection_date": date,
                "country":         country,
                "sequence_length": len(seq_clean),
                "gc_content":      gc_content(seq_clean),
                "n_content_pct":   n_content(seq_clean),
                "description":     rec.description[:120],
                "sequence":        seq_clean,
            })
            accepted_this_batch += 1

        print(f"accepted {accepted_this_batch}  "
              f"(total for lineage: {len(records_out)})")
        start += batch_size
        time.sleep(0.4)

    stats["accepted"] = len(records_out)
    if len(records_out) == 0:
        stats["status"] = "EMPTY"
    elif n_target > 0 and len(records_out) < n_target:
        stats["status"] = f"PARTIAL ({len(records_out)}/{n_target})"

    print(f"    Final: {len(records_out)} accepted  "
          f"| dup_acc={stats['rej_dup_acc']}  "
          f"dup_seq={stats['rej_dup_seq']}  "
          f"quality={stats['rej_quality']}")

    return records_out, stats


def retrieve_all(max_per_lineage=2400, seed=42):
    """Retrieve for all lineages with shared deduplication sets."""
    random.seed(seed)
    seen_accessions = set()
    seen_sequences  = set()

    all_records, all_stats = [], []

    for lineage in LINEAGE_TARGETS:
        records, stats = retrieve_lineage(
            lineage, max_per_lineage,
            seen_accessions, seen_sequences)
        all_records.extend(records)
        all_stats.append(stats)
        time.sleep(1.0)

    return all_records, all_stats


# ═════════════════════════════════════════════════════════════════════════════
# SAVE AND LOG
# ═════════════════════════════════════════════════════════════════════════════

def save_csv(records, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"\n  Saved {len(records):,} records → {path}  ({mb:.1f} MB)")


def save_log(stats, log_path, max_per_lineage, total):
    lines = [
        "="*64,
        "  APBT Study 3 — Sequence Retrieval Log  (v3)",
        f"  Date:              {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Max per lineage:   {max_per_lineage if max_per_lineage > 0 else 'UNLIMITED'}",
        f"  Total retrieved:   {total:,}",
        f"  Sublineages allowed: {ALLOW_SUBLINEAGES}",
        f"  N-content filter:  ≤{N_MAX_PCT}%",
        "="*64, "",
        f"  {'Lineage':<14} {'Max':>7} {'Got':>6} "
        f"{'DupAcc':>7} {'DupSeq':>7} "
        f"{'Quality':>8} {'Status'}",
        "  " + "-"*60,
    ]
    for s in stats:
        lines.append(
            f"  {s['lineage']:<14} {max_per_lineage if max_per_lineage > 0 else '∞':>7} "
            f"{s['accepted']:>6} "
            f"{s['rej_dup_acc']:>7} "
            f"{s['rej_dup_seq']:>7} "
            f"{s['rej_quality']:>8}  "
            f"{s['status']}"
        )
    lines += ["", f"  Total accepted: {total:,}", "="*64]
    with open(log_path,"w",encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve SARS-CoV-2 sequences from NCBI for ML degradation study"
    )
    parser.add_argument("--max-per-lineage", type=int, default=2400,
                        help="Max sequences per lineage (0=unlimited)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: Downloads/study3_sequences_v3_*.csv)")
    parser.add_argument("--no-sublineages", action="store_true",
                        help="Disable sublineage acceptance (exact match only)")
    args = parser.parse_args()

    # Override sublineage setting if flag provided
    global ALLOW_SUBLINEAGES
    if args.no_sublineages:
        ALLOW_SUBLINEAGES = False

    # Generate default output path
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        max_label = f"max{args.max_per_lineage}" if args.max_per_lineage > 0 else "unlimited"
        fname = f"study3_sequences_v3_{max_label}_{ts}.csv"
        args.output = os.path.join(DEFAULT_SAVE_DIR, fname)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    log_path = args.output.replace(".csv", "_log.txt")

    print()
    print("="*64)
    print("  APBT Study 3 — NCBI Sequence Retrieval  (v3)")
    print("  Anjum et al., University of Portsmouth")
    print("="*64)
    print()
    print(f"  Target per lineage: {args.max_per_lineage if args.max_per_lineage > 0 else 'UNLIMITED'}")
    print(f"  Total lineages:     {len(LINEAGE_TARGETS)}")
    print(f"  Quality filters:    length {MIN_SEQ_LENGTH}–{MAX_SEQ_LENGTH} nt  |  N-content ≤{N_MAX_PCT}%")
    print(f"  Sublineages:        {'allowed' if ALLOW_SUBLINEAGES else 'exact match only'}")
    print(f"  Output:             {args.output}")
    print()

    t0 = time.time()

    try:
        all_records, all_stats = retrieve_all(args.max_per_lineage)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

    if not all_records:
        print("\n  No records retrieved.")
        sys.exit(1)

    save_csv(all_records, args.output)
    save_log(all_stats, log_path, args.max_per_lineage, len(all_records))

    elapsed = time.time() - t0

    # Validation summary
    import collections
    lin_counts = collections.Counter(r["lineage"] for r in all_records)
    acc_set    = set(r["accession"] for r in all_records)
    seq_set    = set(r["sequence"]  for r in all_records)
    country_ok = sum(1 for r in all_records if r["country"])

    print()
    print("="*64)
    print("  RETRIEVAL COMPLETE — VALIDATION SUMMARY")
    print("="*64)
    print(f"  Records retrieved    : {len(all_records):,}")
    print(f"  Unique accessions    : {len(acc_set):,}  "
          f"{'✓ no duplicates' if len(acc_set)==len(all_records) else '✗ DUPLICATES REMAIN'}")
    print(f"  Unique sequences     : {len(seq_set):,}  "
          f"{'✓ no duplicates' if len(seq_set)==len(all_records) else '✗ DUPLICATES REMAIN'}")
    print(f"  Country recovered    : {country_ok:,} / {len(all_records):,}")
    print(f"  Time taken           : {elapsed:.0f}s")
    print()
    print(f"  Lineage breakdown:")
    for lin in LINEAGE_TARGETS:
        n = lin_counts.get(lin, 0)
        bar = "█" * (n // 5 if n < 500 else 100)
        pct = n / len(all_records) * 100 if all_records else 0
        print(f"    {lin:<14}  {n:>5}  {pct:>5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()

