"""Temporal holdout sensitivity analysis.

Trains on earlier collection months and tests on the latest months of each
lineage. Reports the clean Random Forest baseline, then Condition D (targeted
label replacement) and Condition E (targeted label injection) under the same
temporal split.

Reads the frozen cohort from data/cohort.csv.xz. Writes its tables, figures
and metadata into reproduced/temporal/.

Run from the repository root:

    python src/temporal_sensitivity.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
import platform
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cohort.csv.xz"
OUT = Path("reproduced") / "temporal"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
K = 4
N_TREES = 100
TARGETED_RATES = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
ATTACK_PAIRS = [
    ("BQ.1", "BA.5"),
    ("BA.5", "BA.2"),
    ("P.1", "B.1.617.2"),
]

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

d_rows = []
for src_name, dst_name in ATTACK_PAIRS:
    src_class = int(le.transform([src_name])[0])
    dst_class = int(le.transform([dst_name])[0])
    source_positions = np.where(y[train_idx] == src_class)[0]
    n_source = len(source_positions)

    rng = np.random.RandomState(SEED)
    ordered_source_positions = rng.permutation(source_positions)

    for rate in TARGETED_RATES:
        m = 0 if rate == 0 else max(1, int(round(n_source * rate)))
        ytr = y[train_idx].copy()
        if m:
            selected = ordered_source_positions[:m]
            ytr[selected] = dst_class

        clf = fit_rf(X[train_idx], ytr)
        pred = clf.predict(X[test_idx])

        d_rows.append({
            "source_lineage": src_name,
            "destination_lineage": dst_name,
            "source_replacement_rate_pct": 100 * rate,
            "source_records_in_temporal_train": n_source,
            "affected_records_m": m,
            "full_temporal_train_burden_pct": 100 * m / len(train_idx),
            "accuracy_pct": 100 * accuracy_score(y[test_idx], pred),
            "macro_f1_pct": 100 * f1_score(
                y[test_idx], pred, average="macro", zero_division=0
            ),
            "source_recall_pct": 100 * recall_score(
                y[test_idx], pred,
                labels=[src_class],
                average="macro",
                zero_division=0,
            ),
        })

d_results = pd.DataFrame(d_rows)
d_results.to_csv(OUT / "temporal_conditionD_results.csv", index=False)

e_rows = []
for src_name, dst_name in ATTACK_PAIRS:
    src_class = int(le.transform([src_name])[0])
    dst_class = int(le.transform([dst_name])[0])
    source_positions = np.where(y[train_idx] == src_class)[0]
    n_source = len(source_positions)

    rng = np.random.RandomState(SEED)
    ordered_source_positions = rng.permutation(source_positions)

    for rate in TARGETED_RATES:
        m = 0 if rate == 0 else max(1, int(round(n_source * rate)))

        if m:
            selected = ordered_source_positions[:m]
            X_aug = np.vstack([
                X[train_idx],
                X[train_idx][selected].copy(),
            ])
            y_aug = np.concatenate([
                y[train_idx],
                np.full(m, dst_class, dtype=y.dtype),
            ])
        else:
            X_aug = X[train_idx]
            y_aug = y[train_idx]

        clf = fit_rf(X_aug, y_aug)
        pred = clf.predict(X[test_idx])

        e_rows.append({
            "source_lineage": src_name,
            "destination_lineage": dst_name,
            "matched_source_rate_pct": 100 * rate,
            "source_records_in_temporal_train": n_source,
            "injected_records_m": m,
            "resulting_temporal_train_size": len(y_aug),
            "injection_burden_pct": 100 * m / len(y_aug),
            "accuracy_pct": 100 * accuracy_score(y[test_idx], pred),
            "macro_f1_pct": 100 * f1_score(
                y[test_idx], pred, average="macro", zero_division=0
            ),
            "source_recall_pct": 100 * recall_score(
                y[test_idx], pred,
                labels=[src_class],
                average="macro",
                zero_division=0,
            ),
        })

e_results = pd.DataFrame(e_rows)
e_results.to_csv(OUT / "temporal_conditionE_results.csv", index=False)

fig = plt.figure(figsize=(8, 5.2))
ax = fig.add_subplot(111)
for (src, dst), g in d_results.groupby(
    ["source_lineage", "destination_lineage"], sort=False
):
    ax.plot(
        g["source_replacement_rate_pct"],
        g["source_recall_pct"],
        marker="o",
        label=f"{src} → {dst}",
    )
ax.set_xlabel("Source-lineage replacement rate (%)")
ax.set_ylabel("Source-lineage recall (%)")
ax.set_ylim(0, 105)
ax.set_title("Temporal holdout sensitivity: Condition D")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "Fig_Temporal_Condition_D.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "Fig_Temporal_Condition_D.pdf", bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(8, 5.2))
ax = fig.add_subplot(111)
for (src, dst), g in e_results.groupby(
    ["source_lineage", "destination_lineage"], sort=False
):
    ax.plot(
        g["injection_burden_pct"],
        g["source_recall_pct"],
        marker="o",
        label=f"{src} → {dst}",
    )
ax.set_xlabel("Injection burden in resulting training set (%)")
ax.set_ylabel("Source-lineage recall (%)")
ax.set_ylim(0, 105)
ax.set_title("Temporal holdout sensitivity: Condition E")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "Fig_Temporal_Condition_E.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "Fig_Temporal_Condition_E.pdf", bbox_inches="tight")
plt.close(fig)

metadata = {
    "dataset": DATA.name,
    "n_total_records": int(len(df)),
    "n_train": int(len(train_idx)),
    "n_test": int(len(test_idx)),
    "n_excluded_year_only": int(len(excluded_idx)),
    "split_rule": (
        "Within each lineage, latest complete collection months assigned "
        "to test until at least 20% of month-resolved records are held out. "
        "Complete months kept intact."
    ),
    "lineage_month_overlap_train_test": 0,
    "classifier": "RandomForestClassifier",
    "n_estimators": N_TREES,
    "random_seed": SEED,
    "feature_dimension": int(X.shape[1]),
    "software": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    },
}
(OUT / "temporal_analysis_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("Temporal analysis complete.")
print("Train:", len(train_idx), "Test:", len(test_idx), "Excluded:", len(excluded_idx))
print("Baseline accuracy:", round(100 * accuracy_score(y[test_idx], baseline_pred), 2))
print("Baseline macro-F1:", round(100 * f1_score(y[test_idx], baseline_pred, average="macro"), 2))
