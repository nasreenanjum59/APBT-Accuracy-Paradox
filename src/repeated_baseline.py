"""Repeated stratified cross-validation on the unmodified study cohort."""
from pathlib import Path
import math
import itertools

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42
K = 4
N_FOLDS = 5

N_REPEATS = 10
REPEAT_SEEDS = list(range(SEED, SEED + N_REPEATS))  # 42--51
T_CRIT_95_DF9 = 2.2621571627409915

TARGET_LINEAGES = ["B.1.1.7", "P.1", "B.1.617.2", "BA.1", "BA.2", "BA.5", "BQ.1"]
BASES = ["A", "C", "G", "T"]
VOCAB = ["".join(p) for p in itertools.product(BASES, repeat=K)]
NFEAT = len(VOCAB)
_CODE = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(BASES):
    _CODE[ord(_b)] = _i



def load_and_filter_data(csv_path):
    print("Loading and filtering data...")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    df = df[df["lineage"].isin(TARGET_LINEAGES)].copy()
    df = df[(df["sequence_length"] >= 29000) & (df["sequence_length"] <= 30200)].copy()
    df = df[df["n_content_pct"] <= 1.0].copy()
    df = df.drop_duplicates(subset=["accession", "sequence"]).reset_index(drop=True)
    print(f"  Final dataset: {len(df)} sequences")
    vc = df["lineage"].value_counts()
    for lin in TARGET_LINEAGES:
        print(f"    {lin}: {vc.get(lin, 0)}")
    return df


def to_vec(seq):
    s = str(seq).upper().replace("N", "A")
    a = _CODE[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]
    a = a[a >= 0]
    if len(a) < K:
        return np.zeros(NFEAT, dtype=np.float32)
    idx = np.zeros(len(a) - K + 1, dtype=np.int64)
    for j in range(K):
        idx = idx * 4 + a[j:len(a) - K + 1 + j]
    v = np.bincount(idx, minlength=NFEAT).astype(np.float32)
    t = v.sum()
    return v / t if t > 0 else v


def build_features(seqs):
    return np.vstack([to_vec(s) for s in seqs])


def get_classifier(clf_name, seed=SEED):
    """Return a classifier. The default random seed is 42."""
    if clf_name == "Random Forest":
        return RandomForestClassifier(100, n_jobs=-1, random_state=seed)
    elif clf_name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=100, random_state=seed,
                                 verbosity=0, eval_metric='mlogloss')
    else:
        return GaussianNB()


def _summary_stats(values):
    """Mean, sample SD and 95% t CI across the 10 repeat-level estimates."""
    a = np.asarray(values, dtype=float)
    mean = float(np.mean(a))
    if len(a) > 1:
        sd = float(np.std(a, ddof=1))
        half = T_CRIT_95_DF9 * sd / math.sqrt(len(a))
    else:
        sd = 0.0
        half = 0.0
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _pooled_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (
        accuracy_score(y_true, y_pred) * 100.0,
        f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0,
    )


def run_repeated_baseline(X_clean, y, le, original_results_abc=None):
    """10 repeated 5-fold CV runs. Seed 42 is reused from the original run when available."""
    print("\n" + "="*78)
    print(f"Clean baseline ({N_REPEATS} seeds: {REPEAT_SEEDS[0]}-{REPEAT_SEEDS[-1]})")
    print("CV partition and stochastic classifier seed vary by repeat")
    print("="*78)

    classifiers = ["Random Forest", "XGBoost", "Naive Bayes"]
    raw_rows = []
    per_lineage_rows = []

    for clf_name in classifiers:
        print(f"\n  {clf_name}")
        for repeat_number, repeat_seed in enumerate(REPEAT_SEEDS, start=1):
            cached = None
            if (repeat_seed == SEED and original_results_abc is not None):
                cached = original_results_abc.get("_baseline_seed42", {}).get(clf_name)

            if cached is not None:
                acc = float(cached["acc"]) * 100.0
                mf1 = float(cached["f1"]) * 100.0
                recalls = np.asarray(cached["recalls"], dtype=float) * 100.0
                print(f"    seed {repeat_seed}: reused original seed-42 baseline")
            else:
                cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=repeat_seed)
                yt, yp = [], []
                for tr, te in cv.split(X_clean, y):
                    clf = get_classifier(clf_name, seed=repeat_seed)
                    weights = compute_sample_weight('balanced', y[tr])
                    clf.fit(X_clean[tr], y[tr], sample_weight=weights)
                    pred = clf.predict(X_clean[te])
                    yt.extend(y[te]); yp.extend(pred)

                acc, mf1 = _pooled_metrics(yt, yp)
                recalls = recall_score(
                    np.asarray(yt), np.asarray(yp), labels=np.arange(len(le.classes_)),
                    average=None, zero_division=0
                ) * 100.0
            row = {
                "classifier": clf_name,
                "repeat": repeat_number,
                "repeat_seed": repeat_seed,
                "accuracy_pct": acc,
                "macro_f1_pct": mf1,
            }
            for i, lin in enumerate(le.classes_):
                row[f"recall_{lin}_pct"] = float(recalls[i])
                per_lineage_rows.append({
                    "classifier": clf_name,
                    "repeat": repeat_number,
                    "repeat_seed": repeat_seed,
                    "lineage": lin,
                    "recall_pct": float(recalls[i]),
                })
            raw_rows.append(row)
            print(f"    seed {repeat_seed}: Acc={acc:6.2f}% | Macro-F1={mf1:6.2f}%")

    raw = pd.DataFrame(raw_rows)
    lineage_raw = pd.DataFrame(per_lineage_rows)

    summary_rows = []
    for clf_name in classifiers:
        g = raw[raw["classifier"] == clf_name]
        a = _summary_stats(g["accuracy_pct"])
        f = _summary_stats(g["macro_f1_pct"])
        summary_rows.append({
            "classifier": clf_name,
            "n_repeats": N_REPEATS,
            "first_seed": REPEAT_SEEDS[0],
            "last_seed": REPEAT_SEEDS[-1],
            "accuracy_mean_pct": a["mean"],
            "accuracy_sd_pct": a["sd"],
            "accuracy_ci95_low_pct": a["ci95_low"],
            "accuracy_ci95_high_pct": a["ci95_high"],
            "macro_f1_mean_pct": f["mean"],
            "macro_f1_sd_pct": f["sd"],
            "macro_f1_ci95_low_pct": f["ci95_low"],
            "macro_f1_ci95_high_pct": f["ci95_high"],
        })

    lineage_summary_rows = []
    for (clf_name, lineage), g in lineage_raw.groupby(["classifier", "lineage"], sort=False):
        s = _summary_stats(g["recall_pct"])
        lineage_summary_rows.append({
            "classifier": clf_name,
            "lineage": lineage,
            "n_repeats": N_REPEATS,
            "recall_mean_pct": s["mean"],
            "recall_sd_pct": s["sd"],
            "recall_ci95_low_pct": s["ci95_low"],
            "recall_ci95_high_pct": s["ci95_high"],
            "recall_min_pct": s["min"],
            "recall_max_pct": s["max"],
        })

    return raw, pd.DataFrame(summary_rows), lineage_raw, pd.DataFrame(lineage_summary_rows)


def build_baseline_paired_comparison(baseline_raw):
    """Paired XGBoost minus Random Forest differences across identical repeat seeds."""
    rows = []
    pivot_acc = baseline_raw.pivot(index="repeat_seed", columns="classifier", values="accuracy_pct")
    pivot_f1 = baseline_raw.pivot(index="repeat_seed", columns="classifier", values="macro_f1_pct")
    for metric_name, pivot in [("accuracy_pct", pivot_acc), ("macro_f1_pct", pivot_f1)]:
        if "XGBoost" not in pivot or "Random Forest" not in pivot:
            continue
        diff = pivot["XGBoost"] - pivot["Random Forest"]
        s = _summary_stats(diff.values)
        rows.append({
            "comparison": "XGBoost - Random Forest",
            "metric": metric_name,
            "n_paired_repeats": len(diff),
            "mean_difference_pp": s["mean"],
            "sd_difference_pp": s["sd"],
            "ci95_low_pp": s["ci95_low"],
            "ci95_high_pp": s["ci95_high"],
            "min_difference_pp": s["min"],
            "max_difference_pp": s["max"],
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = Path("reproduced/baseline")
    out.mkdir(parents=True, exist_ok=True)
    df = load_and_filter_data(root / "data/cohort.csv.xz")
    labels = LabelEncoder().fit(df["lineage"])
    features = build_features(df["sequence"])
    raw, summary, recalls, recall_summary = run_repeated_baseline(
        features, labels.transform(df["lineage"]), labels
    )
    tables = {
        "Baseline_Repeated_Raw": raw,
        "Baseline_Repeated_Summary": summary,
        "Baseline_PerLineage_Recall_Repeated_Raw": recalls,
        "Baseline_PerLineage_Recall_Repeated_Summary": recall_summary,
        "Baseline_XGBoost_vs_RF_Paired_Differences": build_baseline_paired_comparison(raw),
    }
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
