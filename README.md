# APBT Paper 2 cohort provenance

The accession table identifies the 6,675 GenBank records in the frozen analysis dataset. It preserves the original accession strings and annotation fields. The lineage column is the analysis class. The cohort counts table summarises that column.

The retrieval log records 20 July 2026 at 13:20. The time zone was unspecified. The retrieval cap was 2,400 per lineage. BA.2 and BA.5 reached this cap. The class counts describe this sample. XBB.1.5 yielded zero accepted records.

The original CSV contained 6,675 distinct accession strings and 6,675 distinct uppercased sequence strings. Its SHA-256 digest is `8b50dab011fa903ed0f896eae8cc27ea4755d10c3a32ae6ae6fa010044df5295`.

Count the records in each analysis class from the repository root with Python 3:

```python
import csv
from collections import Counter

with open("data/accessions.csv", newline="", encoding="utf-8") as source:
    rows = list(csv.DictReader(source))

assert len(rows) == len({row["accession"] for row in rows}) == 6675
print(Counter(row["lineage"] for row in rows))
```
