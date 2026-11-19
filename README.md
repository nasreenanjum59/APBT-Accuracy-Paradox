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

## Data and retrieval

[Download the frozen sequence CSV](data/cohort.csv.xz). The archive uses XZ compression. After decompression, the CSV should match the SHA-256 checksum above. The accession table is a smaller alternative for metadata analysis.

The [retrieval program](src/retrieve_sequences.py) contains the NCBI queries and filtering rules. It requires Python 3 and Biopython. Set the `NCBI_EMAIL` environment variable to a contact address before running `python src/retrieve_sequences.py --max-per-lineage 2400`. A new search can return different records as GenBank changes.

## Results

The [results tables](results/) contain the repeated baseline evaluations, per-lineage recall, paired classifier comparisons, and aggregate evaluations for Conditions A to C. Accuracy, macro F1 and recall are reported as percentages. Paired differences are in percentage points. The [figures](figures/) show performance under Conditions A to C and F1 by lineage.

## Baseline analyses

Install the Python dependencies with `python -m pip install -r requirements.txt`. From the repository root, run `python src/temporal_baseline.py` for the temporal holdout or `python src/repeated_baseline.py` for repeated stratified cross-validation. Each program writes to its own directory under `reproduced/`.

The [temporal split manifest and results](results/temporal/) identify 3,909 training records, 2,749 test records and 17 excluded records. The [supplementary accession table](supplement/Supplementary_Data_S1_Accession_List.csv) includes the percentage of ambiguous N bases for each record.
