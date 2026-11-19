"""Evaluate the Random Forest baseline on the lineage-specific temporal holdout."""
from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/cohort.csv.xz"
OUT = Path("reproduced/temporal")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
K = 4
N_TREES = 100

df = pd.read_csv(DATA)

BASES = ["A", "C", "G", "T"]
_CODE = np.full(256, -1, dtype=np.int16)
for i, b in enumerate(BASES):
    _CODE[ord(b)] = i

def to_vec(seq):
    s = str(seq).upper().replace("N", "A")
    a = _CODE[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]
    a = a[a >= 0]
    if len(a) < K:
        return np.zeros(4**K, dtype=np.float32)
    idx = np.zeros(len(a) - K + 1, dtype=np.int32)
    for j in range(K):
        idx = idx * 4 + a[j: len(a) - K + 1 + j]
    v = np.bincount(idx, minlength=4**K).astype(np.float32)
    total = v.sum()
    return v / total if total else v

X = np.vstack([to_vec(s) for s in df["sequence"]])
le = LabelEncoder().fit(sorted(df["lineage"].astype(str).unique()))
y = le.transform(df["lineage"].astype(str))

def parse_month(value):
    x = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", x):
        return pd.Period(pd.to_datetime(x), freq="M")
    if re.fullmatch(r"\d{4}-\d{2}", x):
        return pd.Period(x, freq="M")
    if re.fullmatch(r"\d{4}", x):
        return pd.NaT
    try:
        return pd.Period(pd.to_datetime(x, dayfirst=True), freq="M")
    except Exception:
        return pd.NaT

months = df["collection_date"].map(parse_month)

train_idx, test_idx, excluded_idx = [], [], []
split_rows = []
assignment = np.full(len(df), "excluded_year_only_date", dtype=object)

for lineage in le.classes_:
    all_idx = np.where(df["lineage"].astype(str).values == lineage)[0]
    valid_idx = [i for i in all_idx if pd.notna(months.iloc[i])]
    excl_idx = [i for i in all_idx if pd.isna(months.iloc[i])]
    excluded_idx.extend(excl_idx)

    g = pd.DataFrame({
        "idx": valid_idx,
        "month": [months.iloc[i] for i in valid_idx],
    })
    counts = g["month"].value_counts().sort_index()
    target_n = 0.20 * len(g)

    test_months = []
    held = 0
    for month, count in counts.sort_index(ascending=False).items():
        test_months.append(month)
        held += int(count)
        if held >= target_n:
            break

    is_test = g["month"].isin(test_months)
    tr = g.loc[~is_test, "idx"].astype(int).tolist()
    te = g.loc[is_test, "idx"].astype(int).tolist()

    train_idx.extend(tr)
    test_idx.extend(te)
    assignment[tr] = "train"
    assignment[te] = "test"

    split_rows.append({
        "lineage": lineage,
        "total_records": len(all_idx),
        "month_resolved_records": len(valid_idx),
        "train_records": len(tr),
        "test_records": len(te),
        "test_fraction_of_month_resolved_pct": 100 * len(te) / len(valid_idx),
        "excluded_year_only_records": len(excl_idx),
        "first_test_month": str(min(test_months)),
        "last_test_month": str(max(test_months)),
        "last_train_month": str(max(g.loc[~is_test, "month"])) if tr else "",
    })

train_idx = np.asarray(train_idx, dtype=int)
test_idx = np.asarray(test_idx, dtype=int)
excluded_idx = np.asarray(excluded_idx, dtype=int)

split_summary = pd.DataFrame(split_rows)
split_summary.to_csv(OUT / "temporal_holdout_split_summary.csv", index=False)

manifest = df[["accession", "lineage", "collection_date"]].copy()
manifest["parsed_month"] = [str(m) if pd.notna(m) else "" for m in months]
manifest["temporal_assignment"] = assignment
manifest.to_csv(OUT / "temporal_holdout_split_manifest.csv", index=False)

train_pairs = set(zip(
    df.iloc[train_idx]["lineage"].astype(str),
    months.iloc[train_idx].astype(str),
))
test_pairs = set(zip(
    df.iloc[test_idx]["lineage"].astype(str),
    months.iloc[test_idx].astype(str),
))
assert len(train_pairs.intersection(test_pairs)) == 0

def fit_rf(X_train, y_train, seed=SEED):
    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        n_jobs=-1,
        random_state=seed,
    )
    weights = compute_sample_weight("balanced", y_train)
    clf.fit(X_train, y_train, sample_weight=weights)
    return clf

baseline_clf = fit_rf(X[train_idx], y[train_idx])
baseline_pred = baseline_clf.predict(X[test_idx])

pd.DataFrame([{
    "classifier": "Random Forest",
    "seed": SEED,
    "train_records": len(train_idx),
    "test_records": len(test_idx),
    "excluded_year_only_records": len(excluded_idx),
    "accuracy_pct": 100 * accuracy_score(y[test_idx], baseline_pred),
    "macro_f1_pct": 100 * f1_score(
        y[test_idx], baseline_pred, average="macro", zero_division=0
    ),
}]).to_csv(OUT / "temporal_baseline_results.csv", index=False)

baseline_recalls = recall_score(
    y[test_idx],
    baseline_pred,
    labels=np.arange(len(le.classes_)),
    average=None,
    zero_division=0,
)
pd.DataFrame({
    "lineage": le.classes_,
    "temporal_baseline_recall_pct": 100 * baseline_recalls,
}).to_csv(OUT / "temporal_baseline_per_lineage_recall.csv", index=False)

