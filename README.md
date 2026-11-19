# APBT Accuracy Paradox

## Study cohort

The analysis used 6,675 GenBank records from seven lineage groups. The [accession table](data/accessions.csv) identifies each record. Its `lineage` column gives the class used in the analysis.

The [retrieval log](provenance/study3_sequences_v3_max2400_20260720_131521_log.txt) records a run on 20 July 2026 at 13:20 (time zone unspecified). BA.2 and BA.5 each reached the limit of 2,400 records per lineage. The XBB.1.5 query returned no records. The [lineage counts](data/cohort_counts.csv) therefore describe the retained sample.

All 6,675 accession strings were distinct. All 6,675 sequences were also distinct after conversion to uppercase. The SHA-256 checksum of the original sequence CSV is `8b50dab011fa903ed0f896eae8cc27ea4755d10c3a32ae6ae6fa010044df5295`.

## Verify the cohort counts

Run the following Python 3 code from the repository root to compare the accession table with the published counts.

```python
import csv
from collections import Counter

with open("data/accessions.csv", newline="", encoding="utf-8") as source:
    records = list(csv.DictReader(source))

with open("data/cohort_counts.csv", newline="", encoding="utf-8") as source:
    expected = {
        row["lineage"]: int(row["retained_records"])
        for row in csv.DictReader(source)
    }

accessions = {row["accession"] for row in records}
counts = Counter(row["lineage"] for row in records)
assert len(records) == len(accessions) == 6675
assert counts == expected

print(f"{len(records):,} records across {len(counts)} lineage groups")
```
