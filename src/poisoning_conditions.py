"""Classifier degradation under data poisoning: Conditions A to E.

Conditions
    A   random label corruption
    B   spike region substitution
    C   compound, A and B together
    D   targeted label replacement
    E   targeted label injection

Reads the frozen cohort from data/cohort.csv.xz. Writes every table and
figure into reproduced/poisoning_conditions/. Metrics are pooled across the
five held-out folds within a repeat, then averaged over the repeat seeds and
reported with sample SDs and two-sided 95% Student t confidence intervals.

Run from the repository root:

    python src/poisoning_conditions.py
"""

import os
import sys
import time
from pathlib import Path
import csv
import warnings
import itertools
import glob
import math
import platform
import hashlib
from datetime import datetime
warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, f1_score, recall_score, confusion_matrix,
        precision_recall_fscore_support
    )
    from sklearn.utils.class_weight import compute_sample_weight
    import xgboost as xgb
except ImportError as e:
    print(f"\nERROR: Missing dependency: {e}")
    print("Run: pip install pandas numpy scikit-learn xgboost matplotlib seaborn\n")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

SEED = 42
K = 4
N_FOLDS = 5

N_REPEATS = 10
REPEAT_SEEDS = list(range(SEED, SEED + N_REPEATS))  # 42--51
T_CRIT_95_DF9 = 2.2621571627409915

# Run controls.
REGENERATE_ORIGINAL_OUTPUTS = False
RUN_REPEATED_BASELINE = True
RUN_REPEATED_CONDITION_D = True
RUN_REPEATED_ABC = True
RUN_REPEATED_CONDITION_E = True
RUN_HQ_FIGURES = True

# ════════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = str(ROOT / "data" / "cohort.csv.xz")
OUT_DIR = str(Path("reproduced") / "poisoning_conditions")

TARGET_LINEAGES = ["B.1.1.7", "P.1", "B.1.617.2", "BA.1", "BA.2", "BA.5", "BQ.1"]
UNTARGETED_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
TARGETED_RATES = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
ATTACK_PAIRS = [("BQ.1", "BA.5"), ("BA.5", "BA.2"), ("P.1", "B.1.617.2")]

BASES = ["A", "C", "G", "T"]
VOCAB = ["".join(p) for p in itertools.product(BASES, repeat=K)]
NFEAT = len(VOCAB)
_CODE = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(BASES):
    _CODE[ord(_b)] = _i

SPIKE_START = 21563
SPIKE_END = 25384

# Figure defaults.
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.labelsize'] = 11.5
plt.rcParams['axes.titlesize'] = 12.5
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['xtick.labelsize'] = 9.5
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FILTERING
# ═════════════════════════════════════════════════════════════════════════════

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

# ═════════════════════════════════════════════════════════════════════════════
# FEATURES & POISONING
# ═════════════════════════════════════════════════════════════════════════════

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

def mutate_seq_spike_random(seq, n_positions=5):
    if n_positions == 0:
        return seq
    s = list(str(seq))
    hi = min(SPIKE_END, len(s) - 1)
    if hi <= SPIKE_START:
        return seq
    # A SHA-256 seed keeps the substituted positions stable across processes;
    # the built-in hash() is randomised and would change them between reruns.
    digest = hashlib.sha256(str(seq)[:60].encode("utf-8", "ignore")).digest()
    stable_seed = int.from_bytes(digest[:4], byteorder="little", signed=False) % (2**31)
    rng = np.random.RandomState(stable_seed)
    positions = rng.choice(range(SPIKE_START, hi), size=n_positions, replace=False)
    for pos in positions:
        cur = s[pos].upper()
        if cur not in BASES:
            continue
        alt = [b for b in BASES if b != cur]
        s[pos] = rng.choice(alt)
    return "".join(s)

def poison_fold_untargeted(rows, y_clean, rate, condition, n_classes, seed=SEED):
    n = len(rows)
    mut_mask = np.zeros(n, dtype=bool)
    lab = y_clean[rows].copy()
    if rate == 0.0:
        return mut_mask, lab
    k = max(1, int(round(n * rate)))
    picked = np.random.RandomState(seed).choice(n, k, replace=False)
    for i in picked:
        if condition in ("B", "C"):
            mut_mask[i] = True
        if condition in ("A", "C"):
            cur = lab[i]
            lab[i] = np.random.RandomState(seed + rows[i]).choice(
                [c for c in range(n_classes) if c != cur])
    return mut_mask, lab

def poison_fold_targeted(rows, y_clean, source_class, dest_class, rate, seed=SEED):
    lab = y_clean[rows].copy()
    if rate == 0.0:
        return lab
    src_mask = lab == source_class
    src_indices = np.where(src_mask)[0]
    k = max(1, int(round(len(src_indices) * rate)))
    if k == 0:
        return lab
    picked = np.random.RandomState(seed).choice(src_indices, k, replace=False)
    lab[picked] = dest_class
    return lab

def get_classifier(clf_name, seed=SEED):
    """Return a classifier."""
    if clf_name == "Random Forest":
        return RandomForestClassifier(100, n_jobs=-1, random_state=seed)
    elif clf_name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=100, random_state=seed,
                                 verbosity=0, eval_metric='mlogloss')
    else:
        return GaussianNB()

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═════════════════════════════════════════════════════════════════════════════

def run_conditions_abc(X_clean, y, le, df):
    print("\n" + "="*70)
    print("EXPERIMENTS A, B, C: Untargeted Poisoning")
    print("="*70)
    results = {cond: {clf: {} for clf in ["Random Forest", "XGBoost", "Naive Bayes"]}
               for cond in ("A", "B", "C")}
    # Cache the clean seed-42 predictions so the repeated evaluation can reuse
    # them rather than fitting the same model a second time.
    results["_baseline_seed42"] = {}
    classifiers = ["Random Forest", "XGBoost", "Naive Bayes"]
    for condition in ("A", "B", "C"):
        for clf_name in classifiers:
            print(f"\n  {clf_name} - Condition {condition}")
            cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
            for rate in UNTARGETED_RATES:
                yt, yp = [], []
                for tr, te in cv.split(X_clean, y):
                    clf = get_classifier(clf_name)
                    mut_mask, ytr = poison_fold_untargeted(
                        tr, y, rate, condition, len(le.classes_))
                    Xtr = X_clean[tr].copy()
                    if condition in ("B", "C") and np.any(mut_mask):
                        tr_df = df.iloc[tr]
                        seqs_orig = tr_df["sequence"].values
                        seqs_mut = []
                        for idx, (s, m) in enumerate(zip(seqs_orig, mut_mask)):
                            if m:
                                seqs_mut.append(mutate_seq_spike_random(s, n_positions=5))
                        if seqs_mut:
                            X_mut = build_features(seqs_mut)
                            Xtr[mut_mask] = X_mut
                    weights = compute_sample_weight('balanced', ytr)
                    clf.fit(Xtr, ytr, sample_weight=weights)
                    yt.extend(y[te])
                    yp.extend(clf.predict(X_clean[te]))
                acc = accuracy_score(yt, yp)
                f1 = f1_score(yt, yp, average="macro", zero_division=0)
                results[condition][clf_name][rate] = {"acc": acc, "f1": f1}
                if condition == "A" and rate == 0.0:
                    recalls = recall_score(
                        np.asarray(yt), np.asarray(yp),
                        labels=np.arange(len(le.classes_)),
                        average=None, zero_division=0
                    )
                    results["_baseline_seed42"][clf_name] = {
                        "acc": acc, "f1": f1, "recalls": recalls.tolist()
                    }
                print(f"    Rate {rate*100:2.0f}%: Acc={acc*100:.2f}% | F1={f1*100:.2f}%")
    return results

def run_condition_d(X_clean, y, le, df):
    print("\n" + "="*70)
    print("EXPERIMENT D: Targeted Label Corruption")
    print("="*70)
    results = {}
    for src_name, dst_name in ATTACK_PAIRS:
        print(f"\n  {src_name} → {dst_name}")
        src_class = le.transform([src_name])[0]
        dst_class = le.transform([dst_name])[0]
        results[(src_name, dst_name)] = {
            "accuracy": {}, "macro_f1": {}, "source_recall": {},
            # Fold-level cache so the repeated analysis does not refit seed 42.
            "_repeat_seed42": {}
        }
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        for rate in TARGETED_RATES:
            yt, yp = [], []
            src_recalls = []
            source_sizes, affected_counts, train_sizes = [], [], []
            for tr, te in cv.split(X_clean, y):
                clf = get_classifier("Random Forest")
                n_source = int(np.sum(y[tr] == src_class))
                m = 0 if rate == 0.0 else max(1, int(round(n_source * rate)))
                ytr = poison_fold_targeted(tr, y, src_class, dst_class, rate)
                weights = compute_sample_weight('balanced', ytr)
                clf.fit(X_clean[tr], ytr, sample_weight=weights)
                yp_fold = clf.predict(X_clean[te])
                yt.extend(y[te])
                yp.extend(yp_fold)
                rec = recall_score(y[te], yp_fold, labels=[src_class],
                                   average="macro", zero_division=0)
                src_recalls.append(rec)
                source_sizes.append(n_source)
                affected_counts.append(m)
                train_sizes.append(len(tr))
            acc = accuracy_score(yt, yp)
            f1 = f1_score(yt, yp, average="macro", zero_division=0)
            pooled_rec = recall_score(
                np.asarray(yt), np.asarray(yp), labels=[src_class],
                average="macro", zero_division=0
            )
            results[(src_name, dst_name)]["macro_f1"][rate] = f1
            results[(src_name, dst_name)]["accuracy"][rate] = acc
            results[(src_name, dst_name)]["source_recall"][rate] = np.mean(src_recalls)
            results[(src_name, dst_name)]["_repeat_seed42"][rate] = {
                "pooled_source_recall": float(pooled_rec),
                "fold_source_recalls": [float(x) for x in src_recalls],
                "source_sizes": source_sizes,
                "affected_counts": affected_counts,
                "train_sizes": train_sizes,
            }
            print(f"    Rate {rate*100:5.0f}%: Acc={acc*100:6.2f}% | "
                  f"F1={f1*100:6.2f}% | Recall={np.mean(src_recalls)*100:6.1f}%")
    return results

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_conditions_abc(results, fig_dir):
    """Figure 1: Accuracy degradation for Conditions A, B, C."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {'Random Forest': '#D47B6A', 'XGBoost': '#5A8FC2', 'Naive Bayes': '#7A9E5A'}
    markers = {'Random Forest': 'o', 'XGBoost': 's', 'Naive Bayes': '^'}
    for idx, condition in enumerate(("A", "B", "C")):
        ax = axes[idx]
        for clf_name, clf_results in results[condition].items():
            rates = sorted(clf_results.keys())
            accs = [clf_results[r]["acc"]*100 for r in rates]
            ax.plot([r*100 for r in rates], accs,
                   marker=markers.get(clf_name, 'o'),
                   linestyle='-', linewidth=2, markersize=7,
                   label=clf_name, color=colors.get(clf_name, '#1f77b4'))
        cond_labels = {"A": "Label Flip Only", "B": "Spike Mutation Only",
                       "C": "Compound (Label + Spike)"}
        ax.set_title(f"Condition {condition}: {cond_labels[condition]}", fontweight='bold')
        ax.set_xlabel("Poison Rate (%)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_ylim([85, 102])
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='lower left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig1_Conditions_ABC_Degradation.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: Fig1_Conditions_ABC_Degradation.png")

def plot_condition_d(results, fig_dir):
    """Figure 2: Targeted attack performance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (src, dst) in enumerate(ATTACK_PAIRS):
        ax = axes[idx]
        data = results[(src, dst)]
        rates = sorted(data["accuracy"].keys())
        accs = [data["accuracy"][r]*100 for r in rates]
        f1s = [data["macro_f1"][r]*100 for r in rates]
        recs = [data["source_recall"][r]*100 for r in rates]
        ax.plot([r*100 for r in rates], accs, 'o-', linewidth=2.2, markersize=7,
                label='Overall Accuracy', color='#5A8FC2')
        ax.plot([r*100 for r in rates], f1s, 's--', linewidth=2.0, markersize=6,
                label='Macro-F1', color='#7A9E5A')
        ax.plot([r*100 for r in rates], recs, '^-', linewidth=2.2, markersize=7,
                label=f'{src} Recall', color='#D47B6A')
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f"Condition D: {src} → {dst}", fontweight='bold')
        ax.set_xlabel("Poison Rate (% of source lineage)", fontsize=11)
        ax.set_ylabel("Performance (%)", fontsize=11)
        ax.set_ylim([0, 102])
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='center left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig2_Condition_D_Targeted_Attack.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: Fig2_Condition_D_Targeted_Attack.png")

def plot_confusion_matrices_cv(X_clean, y, le, df, fig_dir):
    """Figure 3: Clean vs Condition D (P.1→B.1.617.2 at 70% poisoning)."""
    print("  Generating Fig3 with held-out CV predictions...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    lineage_names = le.classes_

    # 1. Clean Baseline via CV
    y_true_clean, y_pred_clean = [], []
    for tr, te in cv.split(X_clean, y):
        clf = RandomForestClassifier(100, class_weight="balanced",
                                     n_jobs=-1, random_state=SEED)
        clf.fit(X_clean[tr], y[tr])
        y_true_clean.extend(y[te])
        y_pred_clean.extend(clf.predict(X_clean[te]))
    cm_clean = confusion_matrix(y_true_clean, y_pred_clean, normalize='true')
    sns.heatmap(cm_clean, ax=axes[0], annot=True, fmt=".1%", cmap="Blues",
                vmin=0, vmax=1,
                xticklabels=lineage_names, yticklabels=lineage_names,
                annot_kws={'size': 8})
    axes[0].set_title("Clean Baseline (CV Held-Out)", fontweight='bold', fontsize=12)
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("True", fontsize=11)
    axes[0].tick_params(axis='x', rotation=45, labelsize=8)
    axes[0].tick_params(axis='y', rotation=0, labelsize=8)

    # 2. Condition D: P.1 → B.1.617.2 at 70% Poisoning
    y_true_pois, y_pred_pois = [], []
    src_name, dst_name = "P.1", "B.1.617.2"
    src_class = le.transform([src_name])[0]
    dst_class = le.transform([dst_name])[0]
    attack_rate = 0.70

    for tr, te in cv.split(X_clean, y):
        ytr = poison_fold_targeted(tr, y, src_class, dst_class, attack_rate)
        clf = RandomForestClassifier(100, class_weight="balanced",
                                     n_jobs=-1, random_state=SEED)
        clf.fit(X_clean[tr], ytr)
        y_true_pois.extend(y[te])
        y_pred_pois.extend(clf.predict(X_clean[te]))

    cm_pois = confusion_matrix(y_true_pois, y_pred_pois, normalize='true')
    sns.heatmap(cm_pois, ax=axes[1], annot=True, fmt=".1%", cmap="Reds",
                vmin=0, vmax=1,
                xticklabels=lineage_names, yticklabels=lineage_names,
                annot_kws={'size': 8})
    axes[1].set_title(f"Targeted Poisoning: {src_name} → {dst_name}, 70% (CV Held-Out)",
                      fontweight='bold', fontsize=12)
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("True", fontsize=11)
    axes[1].tick_params(axis='x', rotation=45, labelsize=8)
    axes[1].tick_params(axis='y', rotation=0, labelsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig3_Confusion_Matrices.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: Fig3_Confusion_Matrices.png")

def plot_per_lineage_f1_cv(X_clean, y, le, df, fig_dir):
    """Figure 4: Per-lineage F1 scores at 20% poisoning."""
    print("  Generating Fig4 with held-out CV predictions...")
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(le.classes_))
    width = 0.2
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    lineage_names = le.classes_

    # Clean F1 via CV
    y_true_c, y_pred_c = [], []
    for tr, te in cv.split(X_clean, y):
        clf = RandomForestClassifier(100, class_weight="balanced",
                                     n_jobs=-1, random_state=SEED)
        clf.fit(X_clean[tr], y[tr])
        y_true_c.extend(y[te])
        y_pred_c.extend(clf.predict(X_clean[te]))
    _, _, f1_clean, _ = precision_recall_fscore_support(
        y_true_c, y_pred_c, average=None, zero_division=0)

    # Poisoned F1s at 20% via CV
    f1_a, f1_b, f1_c = None, None, None
    for cond_idx, cond in enumerate(["A", "B", "C"]):
        y_true_p, y_pred_p = [], []
        for tr, te in cv.split(X_clean, y):
            df_train = df.iloc[tr].copy().reset_index(drop=True)
            n_p = max(1, int(len(df_train) * 0.20))
            rng_temp = np.random.RandomState(SEED + cond_idx)
            idxs_temp = rng_temp.choice(len(df_train), n_p, replace=False)
            lins_temp = df_train["lineage"].unique().tolist()
            for i in idxs_temp:
                if cond in ("B", "C"):
                    seq = mutate_seq_spike_random(df_train.at[i, "sequence"])
                    df_train.at[i, "sequence"] = seq
                if cond in ("A", "C"):
                    cur = df_train.at[i, "lineage"]
                    df_train.at[i, "lineage"] = rng_temp.choice(
                        [l for l in lins_temp if l != cur])
            X_tr = build_features(df_train["sequence"])
            y_tr = le.transform(df_train["lineage"])
            clf = RandomForestClassifier(100, class_weight="balanced",
                                         n_jobs=-1, random_state=SEED)
            clf.fit(X_tr, y_tr)
            y_true_p.extend(y[te])
            y_pred_p.extend(clf.predict(X_clean[te]))
        _, _, f1_temp, _ = precision_recall_fscore_support(
            y_true_p, y_pred_p, average=None, zero_division=0)
        if cond == "A":
            f1_a = f1_temp
        elif cond == "B":
            f1_b = f1_temp
        else:
            f1_c = f1_temp

    colors = {'Clean': '#2C3E50', 'Cond A': '#D47B6A',
              'Cond B': '#5A8FC2', 'Cond C': '#7A9E5A'}
    datasets = {"Clean": f1_clean*100, "Cond A": f1_a*100,
                "Cond B": f1_b*100, "Cond C": f1_c*100}
    for i, (name, scores) in enumerate(datasets.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, scores, width, label=name,
                     alpha=0.8, color=colors.get(name, '#95a5a6'))
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(lineage_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score (%)", fontsize=11)
    ax.set_title("Per-lineage F1 score at 20% corruption",
                 fontweight='bold', fontsize=12)
    ax.set_ylim([80, 102])
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.2, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Fig4_Per_Lineage_F1.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: Fig4_Per_Lineage_F1.png")

def plot_feature_importance_supplementary(X_clean, y, fig_dir):
    """Supplementary Figure: Top 25 discriminative 4-mers."""
    from matplotlib.patches import Patch
    print("  Generating Supplementary Figure S1 (Feature Importance)...")
    fig, ax = plt.subplots(figsize=(10, 8))
    clf = RandomForestClassifier(100, class_weight="balanced",
                                 n_jobs=-1, random_state=SEED)
    clf.fit(X_clean, y)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[-25:]
    top_kmers = [VOCAB[i] for i in indices]
    top_vals = importances[indices] * 100

    colors = []
    for km in top_kmers:
        gc = (km.count('G') + km.count('C')) / 4.0
        if gc > 0.6:
            colors.append("#5A8FC2")
        elif gc < 0.35:
            colors.append("#D47B6A")
        else:
            colors.append("#7A9E5A")

    bars = ax.barh(range(25), top_vals, color=colors, alpha=0.8)
    for i, (bar, val) in enumerate(zip(bars, top_vals)):
        ax.text(val + 0.02, i, f'{val:.3f}%', va='center', fontsize=7)

    ax.set_yticks(range(25))
    ax.set_yticklabels(top_kmers, family="monospace", fontsize=9)
    ax.set_xlabel("Feature Importance (%)", fontsize=11)
    ax.set_title("Supplementary Fig S1: Top 25 Discriminative 4-mers (Random Forest)",
                 fontweight='bold', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2, axis='x', linestyle='--')

    legend_elements = [
        Patch(facecolor='#D47B6A', label='AT-rich (<35% GC)'),
        Patch(facecolor='#7A9E5A', label='Balanced GC (35-60%)'),
        Patch(facecolor='#5A8FC2', label='GC-rich (>60% GC)')
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "Supplementary_S1_Feature_Importance.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: Supplementary_S1_Feature_Importance.png")

# ═════════════════════════════════════════════════════════════════════════════
# TABLE GENERATION
# ════════════════════════════════════════════════════════════════════════════

def save_results_abc(results, tab_dir):
    for condition in ("A", "B", "C"):
        filename = os.path.join(tab_dir, f"Condition{condition}_Results.csv")
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(["Poison Rate (%)", "Classifier", "Accuracy (%)", "Macro-F1 (%)"])
            for clf_name in results[condition].keys():
                for rate in UNTARGETED_RATES:
                    data = results[condition][clf_name].get(rate, {"acc":0, "f1":0})
                    w.writerow([f"{rate*100:.0f}", clf_name,
                               f"{data['acc']*100:.2f}", f"{data['f1']*100:.2f}"])
        print(f"    Saved: Condition{condition}_Results.csv")

def save_results_d(results, tab_dir):
    for (src, dst), data in results.items():
        filename = os.path.join(tab_dir, f"ConditionD_{src}_to_{dst}.csv")
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(["Poison Rate (%)", "Overall Accuracy (%)",
                       "Macro-F1 (%)", f"{src} Recall (%)"])
            for rate in TARGETED_RATES:
                acc = data["accuracy"].get(rate, 0)
                f1 = data["macro_f1"].get(rate, 0)
                rec = data["source_recall"].get(rate, 0)
                w.writerow([f"{rate*100:.0f}", f"{acc*100:.2f}",
                           f"{f1*100:.2f}", f"{rec*100:.2f}"])
        print(f"    Saved: ConditionD_{src}_to_{dst}.csv")

def generate_summary_tables(results_abc, results_d, le, df, X_clean, y, tab_dir):
    """Generate all summary tables for the paper."""
    print("\n" + "="*70)
    print("GENERATING SUMMARY TABLES FOR PAPER")
    print("="*70)

    # Table 1: Accuracy under untargeted poisoning (Random Forest)
    print("\n  Table 1: Accuracy under untargeted poisoning (Random Forest)")
    table1_path = os.path.join(tab_dir, "Table1_Accuracy_Untargeted.csv")
    with open(table1_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["Poison Rate (%)", "Condition A (Label Flip)",
                    "Condition B (Spike Mutation)", "Condition C (Compound)"])
        for rate in UNTARGETED_RATES:
            acc_a = results_abc["A"]["Random Forest"][rate]["acc"] * 100
            acc_b = results_abc["B"]["Random Forest"][rate]["acc"] * 100
            acc_c = results_abc["C"]["Random Forest"][rate]["acc"] * 100
            w.writerow([f"{rate*100:.0f}", f"{acc_a:.2f}", f"{acc_b:.2f}", f"{acc_c:.2f}"])
    print(f"    Saved: Table1_Accuracy_Untargeted.csv")

    # Table 2: Macro-F1 under untargeted poisoning (Random Forest)
    print("\n  Table 2: Macro-F1 under untargeted poisoning (Random Forest)")
    table2_path = os.path.join(tab_dir, "Table2_MacroF1_Untargeted.csv")
    with open(table2_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["Poison Rate (%)", "Condition A (Label Flip)",
                    "Condition B (Spike Mutation)", "Condition C (Compound)"])
        for rate in UNTARGETED_RATES:
            f1_a = results_abc["A"]["Random Forest"][rate]["f1"] * 100
            f1_b = results_abc["B"]["Random Forest"][rate]["f1"] * 100
            f1_c = results_abc["C"]["Random Forest"][rate]["f1"] * 100
            w.writerow([f"{rate*100:.0f}", f"{f1_a:.2f}", f"{f1_b:.2f}", f"{f1_c:.2f}"])
    print(f"    Saved: Table2_MacroF1_Untargeted.csv")

    # Table 3: Classifier comparison at 30% poisoning
    print("\n  Table 3: Classifier comparison at 30% poisoning")
    table3_path = os.path.join(tab_dir, "Table3_Classifier_Comparison_30pct.csv")
    classifiers = ["Random Forest", "XGBoost", "Naive Bayes"]
    with open(table3_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["Classifier", "Condition A (Label Flip)",
                    "Condition B (Spike Mutation)", "Condition C (Compound)"])
        for clf in classifiers:
            acc_a = results_abc["A"][clf][0.30]["acc"] * 100
            f1_a = results_abc["A"][clf][0.30]["f1"] * 100
            acc_b = results_abc["B"][clf][0.30]["acc"] * 100
            f1_b = results_abc["B"][clf][0.30]["f1"] * 100
            acc_c = results_abc["C"][clf][0.30]["acc"] * 100
            f1_c = results_abc["C"][clf][0.30]["f1"] * 100
            w.writerow([clf,
                       f"{acc_a:.2f} / {f1_a:.2f}",
                       f"{acc_b:.2f} / {f1_b:.2f}",
                       f"{acc_c:.2f} / {f1_c:.2f}"])
    print(f"    Saved: Table3_Classifier_Comparison_30pct.csv")

    # Table 4: Targeted attack full results
    print("\n  Table 4: Targeted attack full results")
    for (src, dst), data in results_d.items():
        table4_path = os.path.join(tab_dir, f"Table4_Targeted_{src}_to_{dst}.csv")
        with open(table4_path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(["Poison Rate (%)", "Overall Accuracy (%)",
                       "Macro-F1 (%)", f"{src} Recall (%)"])
            for rate in TARGETED_RATES:
                acc = data["accuracy"].get(rate, 0) * 100
                f1 = data["macro_f1"].get(rate, 0) * 100
                rec = data["source_recall"].get(rate, 0) * 100
                w.writerow([f"{rate*100:.0f}", f"{acc:.2f}", f"{f1:.2f}", f"{rec:.2f}"])
        print(f"    Saved: Table4_Targeted_{src}_to_{dst}.csv")

    # Table 5: Per-lineage F1 at 20% poisoning
    print("\n  Table 5: Per-lineage F1 at 20% poisoning")

    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    lineage_names = le.classes_

    # Clean F1
    y_true_c, y_pred_c = [], []
    for tr, te in cv.split(X_clean, y):
        clf = RandomForestClassifier(100, class_weight="balanced", n_jobs=-1, random_state=SEED)
        clf.fit(X_clean[tr], y[tr])
        y_true_c.extend(y[te])
        y_pred_c.extend(clf.predict(X_clean[te]))
    _, _, f1_clean, _ = precision_recall_fscore_support(y_true_c, y_pred_c, average=None, zero_division=0)

    # Poisoned F1 at 20%
    f1_a_list, f1_b_list, f1_c_list = [], [], []
    for cond_idx, cond in enumerate(["A", "B", "C"]):
        y_true_p, y_pred_p = [], []
        for tr, te in cv.split(X_clean, y):
            df_train = df.iloc[tr].copy().reset_index(drop=True)
            n_p = max(1, int(len(df_train) * 0.20))
            rng_temp = np.random.RandomState(SEED + cond_idx)
            idxs_temp = rng_temp.choice(len(df_train), n_p, replace=False)
            lins_temp = df_train["lineage"].unique().tolist()
            for i in idxs_temp:
                if cond in ("B", "C"):
                    seq = mutate_seq_spike_random(df_train.at[i, "sequence"])
                    df_train.at[i, "sequence"] = seq
                if cond in ("A", "C"):
                    cur = df_train.at[i, "lineage"]
                    df_train.at[i, "lineage"] = rng_temp.choice([l for l in lins_temp if l != cur])
            X_tr = build_features(df_train["sequence"])
            y_tr = le.transform(df_train["lineage"])
            clf = RandomForestClassifier(100, class_weight="balanced", n_jobs=-1, random_state=SEED)
            clf.fit(X_tr, y_tr)
            y_true_p.extend(y[te])
            y_pred_p.extend(clf.predict(X_clean[te]))
        _, _, f1_temp, _ = precision_recall_fscore_support(y_true_p, y_pred_p, average=None, zero_division=0)
        if cond == "A":
            f1_a_list = f1_temp
        elif cond == "B":
            f1_b_list = f1_temp
        else:
            f1_c_list = f1_temp

    table5_path = os.path.join(tab_dir, "Table5_PerLineage_F1_20pct.csv")
    with open(table5_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["Lineage", "Clean", "Condition A", "Condition B", "Condition C"])
        for i, lin in enumerate(lineage_names):
            w.writerow([lin,
                       f"{f1_clean[i]*100:.0f}",
                       f"{f1_a_list[i]*100:.0f}",
                       f"{f1_b_list[i]*100:.0f}",
                       f"{f1_c_list[i]*100:.0f}"])
    print(f"    Saved: Table5_PerLineage_F1_20pct.csv")

    # Formatted tables for the manuscript are assembled by hand from the CSVs.
    table_text_path = os.path.join(tab_dir, "Tables_Formatted_For_Paper.txt")
    with open(table_text_path, 'w', encoding='utf-8') as f:
        pass

    print(f"  Saved: Tables_Formatted_For_Paper.txt")
    print(f"  All tables saved to: {tab_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# REPEATED EVALUATION
# Multiple seeds, repeated 5-fold CV, sample SDs and 95% t confidence intervals.
# ═════════════════════════════════════════════════════════════════════════════

def _summary_stats(values):
    """Mean, sample SD and 95% t CI across the repeat-level estimates."""
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
    """Clean baseline across the repeat seeds. Seed 42 is reused when cached."""
    print("\n" + "="*78)
    print(f"REVISION STATISTICS: CLEAN BASELINE ({N_REPEATS} seeds: {REPEAT_SEEDS[0]}-{REPEAT_SEEDS[-1]})")
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


def run_repeated_condition_d(X_clean, y, le, original_results_d=None):
    """Repeated evaluation of Condition D. Seed 42 is reused when cached."""
    print("\n" + "="*78)
    print("REVISION STATISTICS: CONDITION D — TARGETED LABEL REPLACEMENT")
    print(f"{N_REPEATS} repeated 5-fold CV runs; seeds {REPEAT_SEEDS[0]}-{REPEAT_SEEDS[-1]}")
    print("="*78)

    raw_rows = []
    fold_rows = []

    for src_name, dst_name in ATTACK_PAIRS:
        src_class = int(le.transform([src_name])[0])
        dst_class = int(le.transform([dst_name])[0])
        print(f"\n  {src_name} -> {dst_name}")

        for rate in TARGETED_RATES:
            for repeat_number, repeat_seed in enumerate(REPEAT_SEEDS, start=1):
                if repeat_seed == SEED and original_results_d is not None:
                    pair_data = original_results_d.get((src_name, dst_name), {})
                    cache = pair_data.get("_repeat_seed42", {}).get(rate)
                    if cache is not None:
                        source_sizes = list(cache["source_sizes"])
                        affected_counts = list(cache["affected_counts"])
                        train_sizes = list(cache["train_sizes"])
                        fold_recalls = list(cache["fold_source_recalls"])
                        for fold_number, (n_source, m, n_train, fold_rec) in enumerate(
                                zip(source_sizes, affected_counts, train_sizes, fold_recalls), start=1):
                            fold_rows.append({
                                "source_lineage": src_name,
                                "destination_lineage": dst_name,
                                "source_replacement_rate_pct": rate * 100,
                                "repeat": repeat_number,
                                "repeat_seed": repeat_seed,
                                "fold": fold_number,
                                "source_records_in_training_fold": n_source,
                                "training_fold_size": n_train,
                                "affected_records_m": m,
                                "full_training_fold_burden_pct": (m / n_train * 100.0),
                                "fold_source_recall_pct": fold_rec * 100.0,
                            })
                        raw_rows.append({
                            "source_lineage": src_name,
                            "destination_lineage": dst_name,
                            "source_replacement_rate_pct": rate * 100,
                            "repeat": repeat_number,
                            "repeat_seed": repeat_seed,
                            "mean_source_records_per_training_fold": float(np.mean(source_sizes)),
                            "mean_affected_records_m": float(np.mean(affected_counts)),
                            "mean_training_fold_size": float(np.mean(train_sizes)),
                            "mean_full_training_fold_burden_pct": float(np.mean(
                                [m/n*100.0 for m, n in zip(affected_counts, train_sizes)]
                            )),
                            "overall_accuracy_pct": float(pair_data["accuracy"][rate]) * 100.0,
                            "overall_macro_f1_pct": float(pair_data["macro_f1"][rate]) * 100.0,
                            "pooled_source_recall_pct": float(cache["pooled_source_recall"]) * 100.0,
                        })
                        print("      seed 42 reused from original Condition D analysis")
                        continue

                cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=repeat_seed)
                yt, yp = [], []
                affected_counts, train_sizes, source_sizes = [], [], []

                for fold_number, (tr, te) in enumerate(cv.split(X_clean, y), start=1):
                    ytr = y[tr].copy()
                    src_positions = np.where(ytr == src_class)[0]
                    n_source = len(src_positions)
                    m = 0 if rate == 0.0 else max(1, int(round(n_source * rate)))
                    if m > 0:
                        # Same repeat/fold seed across rates gives a reproducible ordering.
                        rng = np.random.RandomState(repeat_seed + fold_number * 1009)
                        order = rng.permutation(src_positions)
                        selected = order[:m]
                        ytr[selected] = dst_class

                    clf = get_classifier("Random Forest", seed=repeat_seed)
                    weights = compute_sample_weight('balanced', ytr)
                    clf.fit(X_clean[tr], ytr, sample_weight=weights)
                    pred = clf.predict(X_clean[te])
                    yt.extend(y[te]); yp.extend(pred)

                    fold_rec = recall_score(
                        y[te], pred, labels=[src_class], average="macro", zero_division=0
                    ) * 100.0
                    affected_counts.append(m)
                    train_sizes.append(len(tr))
                    source_sizes.append(n_source)
                    fold_rows.append({
                        "source_lineage": src_name,
                        "destination_lineage": dst_name,
                        "source_replacement_rate_pct": rate * 100,
                        "repeat": repeat_number,
                        "repeat_seed": repeat_seed,
                        "fold": fold_number,
                        "source_records_in_training_fold": n_source,
                        "training_fold_size": len(tr),
                        "affected_records_m": m,
                        "full_training_fold_burden_pct": (m / len(tr) * 100.0),
                        "fold_source_recall_pct": fold_rec,
                    })

                acc, mf1 = _pooled_metrics(yt, yp)
                pooled_rec = recall_score(
                    np.asarray(yt), np.asarray(yp), labels=[src_class],
                    average="macro", zero_division=0
                ) * 100.0
                raw_rows.append({
                    "source_lineage": src_name,
                    "destination_lineage": dst_name,
                    "source_replacement_rate_pct": rate * 100,
                    "repeat": repeat_number,
                    "repeat_seed": repeat_seed,
                    "mean_source_records_per_training_fold": float(np.mean(source_sizes)),
                    "mean_affected_records_m": float(np.mean(affected_counts)),
                    "mean_training_fold_size": float(np.mean(train_sizes)),
                    "mean_full_training_fold_burden_pct": float(np.mean(
                        [m/n*100.0 for m, n in zip(affected_counts, train_sizes)]
                    )),
                    "overall_accuracy_pct": acc,
                    "overall_macro_f1_pct": mf1,
                    "pooled_source_recall_pct": pooled_rec,
                })

            sel = [r for r in raw_rows if r["source_lineage"] == src_name
                   and r["source_replacement_rate_pct"] == rate * 100]
            rv = np.array([r["pooled_source_recall_pct"] for r in sel])
            print(f"    rate {rate*100:>4.0f}% | m={sel[0]['mean_affected_records_m']:>7.1f} | "
                  f"{src_name} recall mean={rv.mean():6.2f}%")

    raw = pd.DataFrame(raw_rows)
    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    keys = ["source_lineage", "destination_lineage", "source_replacement_rate_pct"]
    for key, g in raw.groupby(keys, sort=False):
        a = _summary_stats(g["overall_accuracy_pct"])
        f = _summary_stats(g["overall_macro_f1_pct"])
        r = _summary_stats(g["pooled_source_recall_pct"])
        summary_rows.append({
            "source_lineage": key[0],
            "destination_lineage": key[1],
            "source_replacement_rate_pct": key[2],
            "n_repeats": N_REPEATS,
            "first_seed": REPEAT_SEEDS[0],
            "last_seed": REPEAT_SEEDS[-1],
            "mean_source_records_per_training_fold": float(g["mean_source_records_per_training_fold"].mean()),
            "mean_affected_records_m": float(g["mean_affected_records_m"].mean()),
            "mean_training_fold_size": float(g["mean_training_fold_size"].mean()),
            "mean_full_training_fold_burden_pct": float(g["mean_full_training_fold_burden_pct"].mean()),
            "accuracy_mean_pct": a["mean"], "accuracy_sd_pct": a["sd"],
            "accuracy_ci95_low_pct": a["ci95_low"], "accuracy_ci95_high_pct": a["ci95_high"],
            "macro_f1_mean_pct": f["mean"], "macro_f1_sd_pct": f["sd"],
            "macro_f1_ci95_low_pct": f["ci95_low"], "macro_f1_ci95_high_pct": f["ci95_high"],
            "source_recall_mean_pct": r["mean"], "source_recall_sd_pct": r["sd"],
            "source_recall_ci95_low_pct": r["ci95_low"], "source_recall_ci95_high_pct": r["ci95_high"],
            "source_recall_min_pct": r["min"], "source_recall_max_pct": r["max"],
        })
    return raw, pd.DataFrame(summary_rows), fold_df


def _revision_mutation_vector(seq, original_index, cache):
    """Cache the five position spike substitution vector for one record."""
    key = int(original_index)
    if key not in cache:
        cache[key] = to_vec(mutate_seq_spike_random(seq, n_positions=5))
    return cache[key]


def _revision_untargeted_fold(tr, y, rate, condition, n_classes, repeat_seed):
    """Apply the A/B/C corruption operator with the current repeat seed."""
    return poison_fold_untargeted(
        tr, y, rate, condition, n_classes, seed=repeat_seed
    )


def run_repeated_conditions_abc(X_clean, y, le, df, baseline_raw=None, original_results_abc=None):
    """Repeated evaluation of Conditions A, B and C. Seed 42 is reused when cached."""
    print("\n" + "="*78)
    print("REVISION STATISTICS: CONDITIONS A, B, C")
    print(f"{N_REPEATS} repeated 5-fold CV runs; seeds {REPEAT_SEEDS[0]}-{REPEAT_SEEDS[-1]}")
    print("="*78)
    classifiers = ["Random Forest", "XGBoost", "Naive Bayes"]
    raw_rows = []
    mutation_cache = {}

    # Reuse the repeated clean baseline at rate 0 when available.
    if baseline_raw is not None and len(baseline_raw):
        for condition in ("A", "B", "C"):
            for _, b in baseline_raw.iterrows():
                raw_rows.append({
                    "condition": condition,
                    "poison_rate_pct": 0.0,
                    "classifier": b["classifier"],
                    "repeat": int(b["repeat"]),
                    "repeat_seed": int(b["repeat_seed"]),
                    "accuracy_pct": float(b["accuracy_pct"]),
                    "macro_f1_pct": float(b["macro_f1_pct"]),
                })

    rates_to_run = [r for r in UNTARGETED_RATES if r > 0]
    for condition in ("A", "B", "C"):
        print(f"\n  Condition {condition}")
        for rate in rates_to_run:
            print(f"    rate {rate*100:.0f}%")
            for repeat_number, repeat_seed in enumerate(REPEAT_SEEDS, start=1):
                # Reuse seed 42 rather than refitting an identical
                # condition, rate and classifier.
                if repeat_seed == SEED and original_results_abc is not None:
                    reused = True
                    for clf_name in classifiers:
                        try:
                            m = original_results_abc[condition][clf_name][rate]
                        except (KeyError, TypeError):
                            reused = False
                            break
                    if reused:
                        for clf_name in classifiers:
                            m = original_results_abc[condition][clf_name][rate]
                            raw_rows.append({
                                "condition": condition,
                                "poison_rate_pct": rate * 100,
                                "classifier": clf_name,
                                "repeat": repeat_number,
                                "repeat_seed": repeat_seed,
                                "accuracy_pct": float(m["acc"]) * 100.0,
                                "macro_f1_pct": float(m["f1"]) * 100.0,
                            })
                        print("      seed 42 reused from original analysis")
                        continue

                cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=repeat_seed)
                pooled = {c: {"yt": [], "yp": []} for c in classifiers}
                for tr, te in cv.split(X_clean, y):
                    mut_mask, ytr = _revision_untargeted_fold(
                        tr, y, rate, condition, len(le.classes_), repeat_seed
                    )
                    Xtr = X_clean[tr].copy()
                    if condition in ("B", "C") and np.any(mut_mask):
                        for pos in np.where(mut_mask)[0]:
                            original_index = int(tr[pos])
                            Xtr[pos] = _revision_mutation_vector(
                                df.iloc[original_index]["sequence"],
                                original_index, mutation_cache
                            )
                    weights = compute_sample_weight('balanced', ytr)
                    for clf_name in classifiers:
                        clf = get_classifier(clf_name, seed=repeat_seed)
                        clf.fit(Xtr, ytr, sample_weight=weights)
                        pred = clf.predict(X_clean[te])
                        pooled[clf_name]["yt"].extend(y[te])
                        pooled[clf_name]["yp"].extend(pred)

                for clf_name in classifiers:
                    acc, mf1 = _pooled_metrics(pooled[clf_name]["yt"], pooled[clf_name]["yp"])
                    raw_rows.append({
                        "condition": condition,
                        "poison_rate_pct": rate * 100,
                        "classifier": clf_name,
                        "repeat": repeat_number,
                        "repeat_seed": repeat_seed,
                        "accuracy_pct": acc,
                        "macro_f1_pct": mf1,
                    })

    raw = pd.DataFrame(raw_rows)
    summary_rows = []
    keys = ["condition", "poison_rate_pct", "classifier"]
    for key, g in raw.groupby(keys, sort=False):
        a = _summary_stats(g["accuracy_pct"])
        f = _summary_stats(g["macro_f1_pct"])
        summary_rows.append({
            "condition": key[0], "poison_rate_pct": key[1], "classifier": key[2],
            "n_repeats": len(g),
            "accuracy_mean_pct": a["mean"], "accuracy_sd_pct": a["sd"],
            "accuracy_ci95_low_pct": a["ci95_low"], "accuracy_ci95_high_pct": a["ci95_high"],
            "macro_f1_mean_pct": f["mean"], "macro_f1_sd_pct": f["sd"],
            "macro_f1_ci95_low_pct": f["ci95_low"], "macro_f1_ci95_high_pct": f["ci95_high"],
        })
    return raw, pd.DataFrame(summary_rows)


def build_abc_paired_comparisons(abc_raw):
    """Paired XGBoost minus Random Forest differences for every A/B/C rate."""
    rows = []
    for (condition, rate), g in abc_raw.groupby(["condition", "poison_rate_pct"], sort=False):
        for metric in ("accuracy_pct", "macro_f1_pct"):
            pivot = g.pivot(index="repeat_seed", columns="classifier", values=metric)
            if "XGBoost" not in pivot.columns or "Random Forest" not in pivot.columns:
                continue
            diff = (pivot["XGBoost"] - pivot["Random Forest"]).dropna()
            st = _summary_stats(diff.values)
            rows.append({
                "condition": condition,
                "poison_rate_pct": rate,
                "comparison": "XGBoost - Random Forest",
                "metric": metric,
                "n_paired_repeats": len(diff),
                "mean_difference_pp": st["mean"],
                "sd_difference_pp": st["sd"],
                "ci95_low_pp": st["ci95_low"],
                "ci95_high_pp": st["ci95_high"],
                "min_difference_pp": st["min"],
                "max_difference_pp": st["max"],
            })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# CONDITION E — TARGETED LABEL INJECTION
# Source records are duplicated and the copies carry the destination label, so
# the training fold grows rather than losing correctly labelled records.
# ═════════════════════════════════════════════════════════════════════════════

def _make_injected_training_fold(train_idx, X_clean, y, source_class, dest_class, rate, seed):
    Xtr = X_clean[train_idx]
    ytr = y[train_idx]
    src_positions = np.where(ytr == source_class)[0]
    n_source = len(src_positions)
    m = 0 if rate == 0.0 else max(1, int(round(n_source * rate)))
    if m == 0:
        return Xtr.copy(), ytr.copy(), 0, n_source, len(train_idx), len(train_idx)
    rng = np.random.RandomState(seed)
    selected = rng.choice(src_positions, size=m, replace=False)
    X_aug = np.vstack([Xtr, Xtr[selected].copy()])
    y_aug = np.concatenate([ytr, np.full(m, dest_class, dtype=ytr.dtype)])
    return X_aug, y_aug, m, n_source, len(train_idx), len(y_aug)


def run_repeated_condition_e(X_clean, y, le):
    print("\n" + "="*78)
    print("REVISION STATISTICS: CONDITION E — TARGETED LABEL INJECTION")
    print("="*78)
    rows = []
    for src_name, dst_name in ATTACK_PAIRS:
        sc = int(le.transform([src_name])[0]); dc = int(le.transform([dst_name])[0])
        for rate in TARGETED_RATES:
            for repeat_number, repeat_seed in enumerate(REPEAT_SEEDS, start=1):
                cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=repeat_seed)
                yt, yp, ms, burdens, src_sizes = [], [], [], [], []
                for tr, te in cv.split(X_clean, y):
                    X_aug, y_aug, m, n_source, n_orig, n_result = _make_injected_training_fold(
                        tr, X_clean, y, sc, dc, rate, repeat_seed
                    )
                    clf = get_classifier("Random Forest", seed=repeat_seed)
                    weights = compute_sample_weight('balanced', y_aug)
                    clf.fit(X_aug, y_aug, sample_weight=weights)
                    pred = clf.predict(X_clean[te])
                    yt.extend(y[te]); yp.extend(pred)
                    ms.append(m); src_sizes.append(n_source)
                    burdens.append(m / n_result * 100.0 if n_result else 0.0)
                acc, mf1 = _pooled_metrics(yt, yp)
                rec = recall_score(np.asarray(yt), np.asarray(yp), labels=[sc],
                                   average="macro", zero_division=0) * 100.0
                rows.append({
                    "source_lineage": src_name, "destination_lineage": dst_name,
                    "matched_condition_d_source_rate_pct": rate * 100,
                    "repeat": repeat_number, "repeat_seed": repeat_seed,
                    "mean_source_records_per_training_fold": float(np.mean(src_sizes)),
                    "mean_injected_records_m": float(np.mean(ms)),
                    "mean_injection_burden_pct": float(np.mean(burdens)),
                    "overall_accuracy_pct": acc, "overall_macro_f1_pct": mf1,
                    "pooled_source_recall_pct": rec,
                })
    raw = pd.DataFrame(rows)
    summary_rows = []
    keys = ["source_lineage", "destination_lineage", "matched_condition_d_source_rate_pct"]
    for key, g in raw.groupby(keys, sort=False):
        a = _summary_stats(g["overall_accuracy_pct"])
        f = _summary_stats(g["overall_macro_f1_pct"])
        r = _summary_stats(g["pooled_source_recall_pct"])
        summary_rows.append({
            "source_lineage": key[0], "destination_lineage": key[1],
            "matched_condition_d_source_rate_pct": key[2],
            "n_repeats": len(g),
            "mean_source_records_per_training_fold": float(g["mean_source_records_per_training_fold"].mean()),
            "mean_injected_records_m": float(g["mean_injected_records_m"].mean()),
            "mean_injection_burden_pct": float(g["mean_injection_burden_pct"].mean()),
            "accuracy_mean_pct": a["mean"], "accuracy_sd_pct": a["sd"],
            "accuracy_ci95_low_pct": a["ci95_low"], "accuracy_ci95_high_pct": a["ci95_high"],
            "macro_f1_mean_pct": f["mean"], "macro_f1_sd_pct": f["sd"],
            "macro_f1_ci95_low_pct": f["ci95_low"], "macro_f1_ci95_high_pct": f["ci95_high"],
            "source_recall_mean_pct": r["mean"], "source_recall_sd_pct": r["sd"],
            "source_recall_ci95_low_pct": r["ci95_low"], "source_recall_ci95_high_pct": r["ci95_high"],
            "source_recall_min_pct": r["min"], "source_recall_max_pct": r["max"],
        })
    return raw, pd.DataFrame(summary_rows)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# Saved as PNG at 600 dpi and as vector PDF.
# ═════════════════════════════════════════════════════════════════════════════

HQ_ORANGE = '#D47B6A'
HQ_BLUE = '#5A8FC2'
HQ_GREEN = '#7A9E5A'
HQ_DARK = '#2C3E50'
HQ_GRID = '#D9D9D9'


def _polish_ax(ax, grid=False):
    ax.set_facecolor('white')
    if grid:
        ax.grid(True, axis='y', color=HQ_GRID, linestyle='--', linewidth=0.55, alpha=0.45, zorder=0)
    else:
        ax.grid(False)
    for side in ('left', 'bottom', 'top', 'right'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('black')
        ax.spines[side].set_linewidth(1.15)
    ax.tick_params(axis='both', colors='black', width=0.95, length=4)


def _add_panel_label(ax, label):
    # Panel tag sits just outside the upper-left border so it cannot be hidden by data.
    ax.text(-0.085, 1.075, label, transform=ax.transAxes, ha='left', va='top',
            fontsize=13.0, fontweight='bold', color='black', clip_on=False, zorder=20,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.18', linewidth=1.0))


def _save_hq(fig, fig_dir, base_name):
    fig.patch.set_facecolor('white')
    png = os.path.join(fig_dir, base_name + '.png')
    pdf = os.path.join(fig_dir, base_name + '.pdf')
    fig.savefig(png, dpi=600, bbox_inches='tight', facecolor='white', pad_inches=0.08)
    fig.savefig(pdf, bbox_inches='tight', facecolor='white', pad_inches=0.08)
    plt.close(fig)
    print(f"  Saved HQ: {os.path.basename(png)} and {os.path.basename(pdf)}")


def plot_conditions_abc_hq(results, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3), constrained_layout=False)
    colors = {'Random Forest': HQ_ORANGE, 'XGBoost': HQ_BLUE, 'Naive Bayes': HQ_GREEN}
    markers = {'Random Forest': 'o', 'XGBoost': 's', 'Naive Bayes': '^'}
    cond_labels = {"A": "Label corruption", "B": "Spike-region substitution",
                   "C": "Compound corruption"}
    for idx, condition in enumerate(("A", "B", "C")):
        ax = axes[idx]
        _polish_ax(ax, grid=False)
        _add_panel_label(ax, chr(65 + idx))
        for clf_name, clf_results in results[condition].items():
            rates = sorted(clf_results.keys())
            accs = [clf_results[r]["acc"]*100 for r in rates]
            ax.plot([r*100 for r in rates], accs,
                    marker=markers[clf_name], markersize=7.2, linewidth=2.25,
                    markeredgecolor='black', markeredgewidth=0.8,
                    color=colors[clf_name], label=clf_name, zorder=3)
        ax.set_title(f"Condition {condition}: {cond_labels[condition]}", fontweight='bold', pad=10)
        ax.set_xlabel("Poison rate (%)")
        if idx == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(85, 101)
        ax.set_xticks([0, 5, 10, 20, 30])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22, wspace=0.20)
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    _save_hq(fig, fig_dir, 'Fig1_Conditions_ABC_Degradation')


def plot_condition_d_hq(results, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3), constrained_layout=False)
    for idx, (src, dst) in enumerate(ATTACK_PAIRS):
        ax = axes[idx]
        _polish_ax(ax, grid=False)
        _add_panel_label(ax, chr(65 + idx))
        data = results[(src, dst)]
        rates = sorted(data['accuracy'].keys())
        x = [r*100 for r in rates]
        series = [
            ('Overall accuracy', [data['accuracy'][r]*100 for r in rates], HQ_BLUE, 'o', '-'),
            ('Macro-F1', [data['macro_f1'][r]*100 for r in rates], HQ_GREEN, 's', '--'),
            ('Source-lineage recall', [data['source_recall'][r]*100 for r in rates], HQ_ORANGE, '^', '-'),
        ]
        for label, vals, color, marker, ls in series:
            ax.plot(x, vals, marker=marker, linestyle=ls, linewidth=2.25,
                    markersize=7.0, markeredgecolor='black', markeredgewidth=0.8,
                    color=color, label=label, zorder=3)
        ax.set_title(f"{src} → {dst}", fontweight='bold', pad=10)
        ax.set_xlabel("Source-lineage replacement rate (%)")
        if idx == 0:
            ax.set_ylabel("Performance (%)")
        ax.set_ylim(0, 101)
        ax.set_xticks([0, 10, 20, 30, 50, 70, 100])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22, wspace=0.20)
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    _save_hq(fig, fig_dir, 'Fig2_Condition_D_Targeted_Attack')


def plot_confusion_matrices_hq(X_clean, y, le, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    lineage_names = le.classes_
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    yt0, yp0 = [], []
    for tr, te in cv.split(X_clean, y):
        clf = RandomForestClassifier(100, class_weight='balanced', n_jobs=-1, random_state=SEED)
        clf.fit(X_clean[tr], y[tr])
        yt0.extend(y[te]); yp0.extend(clf.predict(X_clean[te]))
    cm0 = confusion_matrix(yt0, yp0, normalize='true')

    src = int(le.transform(['P.1'])[0]); dst = int(le.transform(['B.1.617.2'])[0])
    yt1, yp1 = [], []
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X_clean, y):
        ytr = poison_fold_targeted(tr, y, src, dst, 0.70)
        clf = RandomForestClassifier(100, class_weight='balanced', n_jobs=-1, random_state=SEED)
        clf.fit(X_clean[tr], ytr)
        yt1.extend(y[te]); yp1.extend(clf.predict(X_clean[te]))
    cm1 = confusion_matrix(yt1, yp1, normalize='true')

    labels = ['A', 'B']
    for i, (ax, cm, cmap, title) in enumerate([
        (axes[0], cm0, 'Blues', 'Clean baseline'),
        (axes[1], cm1, 'Oranges', 'P.1 → B.1.617.2 at 70% replacement'),
    ]):
        _add_panel_label(ax, labels[i])
        sns.heatmap(cm, ax=ax, annot=True, fmt='.1%', cmap=cmap, vmin=0, vmax=1,
                    xticklabels=lineage_names, yticklabels=lineage_names,
                    linewidths=0.45, linecolor='black', cbar=True,
                    annot_kws={'size': 8.5, 'weight': 'bold'})
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted lineage')
        ax.set_ylabel('True lineage')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(0.9)
    _save_hq(fig, fig_dir, 'Fig3_Confusion_Matrices')


def plot_per_lineage_f1_hq(X_clean, y, le, df, fig_dir):
    lineage_names = le.classes_
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    yt, yp = [], []
    for tr, te in cv.split(X_clean, y):
        clf = RandomForestClassifier(100, class_weight='balanced', n_jobs=-1, random_state=SEED)
        clf.fit(X_clean[tr], y[tr]); yt.extend(y[te]); yp.extend(clf.predict(X_clean[te]))
    _, _, f1_clean, _ = precision_recall_fscore_support(yt, yp, average=None, zero_division=0)

    cond_scores = {}
    for cond_idx, cond in enumerate(['A','B','C']):
        yt, yp = [], []
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        for tr, te in cv.split(X_clean, y):
            df_train = df.iloc[tr].copy().reset_index(drop=True)
            n_p = max(1, int(len(df_train)*0.20))
            rng = np.random.RandomState(SEED + cond_idx)
            idxs = rng.choice(len(df_train), n_p, replace=False)
            lins = df_train['lineage'].unique().tolist()
            for i in idxs:
                if cond in ('B','C'):
                    df_train.at[i,'sequence'] = mutate_seq_spike_random(df_train.at[i,'sequence'])
                if cond in ('A','C'):
                    cur = df_train.at[i,'lineage']
                    df_train.at[i,'lineage'] = rng.choice([l for l in lins if l != cur])
            Xtr = build_features(df_train['sequence'])
            ytr = le.transform(df_train['lineage'])
            clf = RandomForestClassifier(100, class_weight='balanced', n_jobs=-1, random_state=SEED)
            clf.fit(Xtr, ytr); yt.extend(y[te]); yp.extend(clf.predict(X_clean[te]))
        _, _, fs, _ = precision_recall_fscore_support(yt, yp, average=None, zero_division=0)
        cond_scores[cond] = fs * 100

    fig, ax = plt.subplots(figsize=(14.5, 6.3), constrained_layout=True)
    _polish_ax(ax, grid=False)
    x = np.arange(len(lineage_names)); width = 0.19
    data = [
        ('Clean', f1_clean*100, HQ_DARK),
        ('Condition A', cond_scores['A'], HQ_ORANGE),
        ('Condition B', cond_scores['B'], HQ_BLUE),
        ('Condition C', cond_scores['C'], HQ_GREEN),
    ]
    for i, (label, vals, color) in enumerate(data):
        bars = ax.bar(x + (i-1.5)*width, vals, width, label=label,
                      color=color, edgecolor='black', linewidth=0.75, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.25, f'{val:.0f}',
                    ha='center', va='bottom', fontsize=7.2, color='black')
    ax.set_xticks(x); ax.set_xticklabels(lineage_names, rotation=35, ha='right')
    ax.set_ylabel('F1 score (%)')
    ax.set_ylim(80, 102)
    ax.set_title('Per-lineage F1 under clean training and 20% corruption', fontweight='bold', pad=10)
    ax.legend(ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.24))
    _save_hq(fig, fig_dir, 'Fig4_Per_Lineage_F1')


def plot_feature_importance_hq(X_clean, y, fig_dir):
    from matplotlib.patches import Patch
    clf = RandomForestClassifier(100, class_weight='balanced', n_jobs=-1, random_state=SEED)
    clf.fit(X_clean, y)
    indices = np.argsort(clf.feature_importances_)[-25:]
    kmers = [VOCAB[i] for i in indices]
    vals = clf.feature_importances_[indices]*100
    colors = []
    for km in kmers:
        gc = (km.count('G') + km.count('C')) / 4.0
        colors.append(HQ_BLUE if gc > 0.6 else HQ_ORANGE if gc < 0.35 else HQ_GREEN)
    fig, ax = plt.subplots(figsize=(10.2, 8.2), constrained_layout=True)
    _polish_ax(ax, grid=False)
    bars = ax.barh(np.arange(25), vals, color=colors, edgecolor='black', linewidth=0.7, zorder=3)
    ax.set_yticks(np.arange(25)); ax.set_yticklabels(kmers, family='monospace')
    ax.set_xlabel('Feature importance (%)')
    ax.set_title('Top 25 discriminative 4-mers', fontweight='bold', pad=10)
    for bar, val in zip(bars, vals):
        ax.text(val + max(vals)*0.01, bar.get_y()+bar.get_height()/2,
                f'{val:.3f}%', va='center', fontsize=7.5)
    ax.legend(handles=[
        Patch(facecolor=HQ_ORANGE, edgecolor='black', label='AT-rich (<35% GC)'),
        Patch(facecolor=HQ_GREEN, edgecolor='black', label='Balanced GC (35-60%)'),
        Patch(facecolor=HQ_BLUE, edgecolor='black', label='GC-rich (>60% GC)')],
        loc='lower right', frameon=False)
    _save_hq(fig, fig_dir, 'Supplementary_S1_Feature_Importance')


def plot_conditions_abc_repeated_hq(summary, fig_dir):
    """Figure 1 from the repeated-CV means, with 95% CI error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3), constrained_layout=False)
    colors = {'Random Forest': HQ_ORANGE, 'XGBoost': HQ_BLUE, 'Naive Bayes': HQ_GREEN}
    markers = {'Random Forest': 'o', 'XGBoost': 's', 'Naive Bayes': '^'}
    cond_labels = {"A": "Label corruption", "B": "Spike-region substitution",
                   "C": "Compound corruption"}
    for idx, condition in enumerate(("A", "B", "C")):
        ax = axes[idx]
        _polish_ax(ax, grid=False)
        _add_panel_label(ax, chr(65 + idx))
        for clf_name in ("Random Forest", "XGBoost", "Naive Bayes"):
            g = summary[(summary["condition"] == condition) &
                        (summary["classifier"] == clf_name)].sort_values("poison_rate_pct")
            if g.empty:
                continue
            x = g["poison_rate_pct"].to_numpy(float)
            y = g["accuracy_mean_pct"].to_numpy(float)
            lo = g["accuracy_ci95_low_pct"].to_numpy(float)
            hi = g["accuracy_ci95_high_pct"].to_numpy(float)
            yerr = np.vstack([y-lo, hi-y])
            ax.errorbar(x, y, yerr=yerr, marker=markers[clf_name], markersize=7.2,
                        linewidth=2.25, capsize=3.2, elinewidth=1.0,
                        markeredgecolor='black', markeredgewidth=0.8,
                        color=colors[clf_name], label=clf_name, zorder=3)
        ax.set_title(f"Condition {condition}: {cond_labels[condition]}", fontweight='bold', pad=10)
        ax.set_xlabel("Poison rate (%)")
        if idx == 0:
            ax.set_ylabel("Accuracy, mean across repeats (%)")
        ax.set_ylim(85, 101)
        ax.set_xticks([0, 5, 10, 20, 30])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22, wspace=0.20)
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    _save_hq(fig, fig_dir, 'Fig1_Conditions_ABC_Degradation')


def plot_condition_d_repeated_hq(summary, fig_dir):
    """Figure 2 from the repeated-CV means, with 95% CI error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3), constrained_layout=False)
    for idx, (src, dst) in enumerate(ATTACK_PAIRS):
        ax = axes[idx]
        _polish_ax(ax, grid=False)
        _add_panel_label(ax, chr(65 + idx))
        g = summary[(summary["source_lineage"] == src) &
                    (summary["destination_lineage"] == dst)].sort_values("source_replacement_rate_pct")
        x = g["source_replacement_rate_pct"].to_numpy(float)
        specs = [
            ('Overall accuracy', 'accuracy_mean_pct', 'accuracy_ci95_low_pct', 'accuracy_ci95_high_pct', HQ_BLUE, 'o', '-'),
            ('Macro-F1', 'macro_f1_mean_pct', 'macro_f1_ci95_low_pct', 'macro_f1_ci95_high_pct', HQ_GREEN, 's', '--'),
            ('Source-lineage recall', 'source_recall_mean_pct', 'source_recall_ci95_low_pct', 'source_recall_ci95_high_pct', HQ_ORANGE, '^', '-'),
        ]
        for label, mcol, lcol, hcol, color, marker, ls in specs:
            y = g[mcol].to_numpy(float); lo = g[lcol].to_numpy(float); hi = g[hcol].to_numpy(float)
            yerr = np.vstack([y-lo, hi-y])
            ax.errorbar(x, y, yerr=yerr, marker=marker, linestyle=ls, linewidth=2.25,
                        markersize=7.0, capsize=3.0, elinewidth=1.0,
                        markeredgecolor='black', markeredgewidth=0.8,
                        color=color, label=label, zorder=3)
        ax.set_title(f"{src} → {dst}", fontweight='bold', pad=10)
        ax.set_xlabel("Source-lineage replacement rate (%)")
        if idx == 0:
            ax.set_ylabel("Performance, mean across repeats (%)")
        ax.set_ylim(0, 101)
        ax.set_xticks([0, 10, 20, 30, 50, 70, 100])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22, wspace=0.20)
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    _save_hq(fig, fig_dir, 'Fig2_Condition_D_Targeted_Attack')



def plot_condition_e_repeated_hq(summary, fig_dir):
    """Condition E: targeted label injection, repeated-CV means with 95% CIs."""
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3), constrained_layout=False)
    for idx, (src, dst) in enumerate(ATTACK_PAIRS):
        ax = axes[idx]
        _polish_ax(ax, grid=False)
        _add_panel_label(ax, chr(65 + idx))
        g = summary[(summary["source_lineage"] == src) &
                    (summary["destination_lineage"] == dst)].sort_values("mean_injection_burden_pct")
        if g.empty:
            ax.set_visible(False)
            continue
        x = g["mean_injection_burden_pct"].to_numpy(float)
        specs = [
            ('Overall accuracy', 'accuracy_mean_pct', 'accuracy_ci95_low_pct', 'accuracy_ci95_high_pct', HQ_BLUE, 'o', '-'),
            ('Macro-F1', 'macro_f1_mean_pct', 'macro_f1_ci95_low_pct', 'macro_f1_ci95_high_pct', HQ_GREEN, 's', '--'),
            ('Source-lineage recall', 'source_recall_mean_pct', 'source_recall_ci95_low_pct', 'source_recall_ci95_high_pct', HQ_ORANGE, '^', '-'),
        ]
        for label, mcol, lcol, hcol, color, marker, ls in specs:
            yv = g[mcol].to_numpy(float)
            lo = g[lcol].to_numpy(float)
            hi = g[hcol].to_numpy(float)
            yerr = np.vstack([yv - lo, hi - yv])
            ax.errorbar(x, yv, yerr=yerr, marker=marker, linestyle=ls, linewidth=2.25,
                        markersize=7.0, capsize=3.0, elinewidth=1.0,
                        markeredgecolor='black', markeredgewidth=0.8,
                        color=color, label=label, zorder=3)
        ax.set_title(f"{src} → {dst}", fontweight='bold', pad=10)
        ax.set_xlabel("Injection burden in resulting training fold (%)")
        if idx == 0:
            ax.set_ylabel("Performance, mean across repeats (%)")
        ax.set_ylim(0, 101)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22, wspace=0.20)
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02))
    _save_hq(fig, fig_dir, 'Fig5_Condition_E_Targeted_Label_Injection')

def write_revision_config(path, df, X_clean):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('SCIENTIFIC REPORTS REVISION — REPEATED EVALUATION CONFIGURATION\n')
        f.write('='*72 + '\n')
        f.write(f'Run time: {datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'Dataset: {CSV_PATH}\n')
        f.write(f'Filtered dataset size: {len(df)}\n')
        f.write(f'Feature dimension: {X_clean.shape[1]}\n')
        f.write(f'CV: {N_FOLDS}-fold stratified, repeated {N_REPEATS} times\n')
        f.write(f'Repeat seeds: {REPEAT_SEEDS}\n')
        f.write('Within each repeat, held-out predictions are pooled across the five folds.\n')
        f.write('Reported uncertainty: sample SD and two-sided 95% Student t CI across repeat-level estimates.\n')
        f.write('Condition D burden: affected records / original full training fold.\n')
        f.write('Condition E burden: injected records / resulting full training fold.\n')
        f.write('Spike substitutions use a deterministic SHA-256-derived seed for cross-process reproducibility.\n')
        f.write('\nSoftware versions:\n')
        f.write(f'  Python: {platform.python_version()}\n')
        f.write(f'  NumPy: {np.__version__}\n')
        f.write(f'  pandas: {pd.__version__}\n')
        f.write(f'  scikit-learn: {__import__("sklearn").__version__}\n')
        f.write(f'  xgboost: {xgb.__version__}\n')
        f.write(f'  matplotlib: {matplotlib.__version__}\n')
        f.write(f'  seaborn: {sns.__version__}\n')
        f.write('\nRun switches:\n')
        f.write(f'  REGENERATE_ORIGINAL_OUTPUTS={REGENERATE_ORIGINAL_OUTPUTS}\n')
        f.write(f'  RUN_REPEATED_BASELINE={RUN_REPEATED_BASELINE}\n')
        f.write(f'  RUN_REPEATED_CONDITION_D={RUN_REPEATED_CONDITION_D}\n')
        f.write(f'  RUN_REPEATED_ABC={RUN_REPEATED_ABC}\n')
        f.write(f'  RUN_REPEATED_CONDITION_E={RUN_REPEATED_CONDITION_E}\n')
        f.write(f'  RUN_HQ_FIGURES={RUN_HQ_FIGURES}\n')



def preflight_plotting_check(out_dir):
    """Fail fast on plotting/save problems before the long repeated-CV run."""
    test_path = os.path.join(out_dir, "__plotting_preflight__.png")
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    _polish_ax(ax, grid=False)
    _add_panel_label(ax, "A")
    ax.plot([0, 1], [0, 1], color=HQ_BLUE, marker='o',
            markeredgecolor='black', markeredgewidth=0.8)
    ax.set_title("BQ.1 → BA.5")
    ax.set_xlabel("Preflight x")
    ax.set_ylabel("Preflight y")
    fig.savefig(test_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    if os.path.exists(test_path):
        os.remove(test_path)
    print("Plotting preflight passed.")


def build_condition_d_e_comparison(d_summary, e_summary):
    """Create a descriptive D-versus-E table using each condition's own burden definition."""
    if d_summary is None or e_summary is None or d_summary.empty or e_summary.empty:
        return pd.DataFrame()
    d = d_summary.copy().rename(columns={
        'source_replacement_rate_pct': 'matched_source_rate_pct',
        'mean_affected_records_m': 'condition_d_mean_affected_m',
        'mean_full_training_fold_burden_pct': 'condition_d_full_fold_burden_pct',
        'source_recall_mean_pct': 'condition_d_source_recall_mean_pct',
        'source_recall_sd_pct': 'condition_d_source_recall_sd_pct',
        'source_recall_ci95_low_pct': 'condition_d_source_recall_ci95_low_pct',
        'source_recall_ci95_high_pct': 'condition_d_source_recall_ci95_high_pct',
    })
    e = e_summary.copy().rename(columns={
        'matched_condition_d_source_rate_pct': 'matched_source_rate_pct',
        'mean_injected_records_m': 'condition_e_mean_injected_m',
        'mean_injection_burden_pct': 'condition_e_injection_burden_pct',
        'source_recall_mean_pct': 'condition_e_source_recall_mean_pct',
        'source_recall_sd_pct': 'condition_e_source_recall_sd_pct',
        'source_recall_ci95_low_pct': 'condition_e_source_recall_ci95_low_pct',
        'source_recall_ci95_high_pct': 'condition_e_source_recall_ci95_high_pct',
    })
    dcols = ['source_lineage','destination_lineage','matched_source_rate_pct',
             'condition_d_mean_affected_m','condition_d_full_fold_burden_pct',
             'condition_d_source_recall_mean_pct','condition_d_source_recall_sd_pct',
             'condition_d_source_recall_ci95_low_pct','condition_d_source_recall_ci95_high_pct']
    ecols = ['source_lineage','destination_lineage','matched_source_rate_pct',
             'condition_e_mean_injected_m','condition_e_injection_burden_pct',
             'condition_e_source_recall_mean_pct','condition_e_source_recall_sd_pct',
             'condition_e_source_recall_ci95_low_pct','condition_e_source_recall_ci95_high_pct']
    out = d[dcols].merge(e[ecols], on=['source_lineage','destination_lineage','matched_source_rate_pct'], how='inner')
    out['source_recall_E_minus_D_pp'] = (out['condition_e_source_recall_mean_pct'] -
                                         out['condition_d_source_recall_mean_pct'])
    return out

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*70)
    print("APBT STUDY 3 — ML DEGRADATION EXPERIMENT")
    print("ORIGINAL ANALYSIS + SCIENTIFIC REPORTS REVISION STATISTICS")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results output root: {OUT_DIR}")
    print("Original submitted files are protected unless REGENERATE_ORIGINAL_OUTPUTS=True.")

    os.makedirs(OUT_DIR, exist_ok=True)
    # Every output goes in one folder.
    fig_dir = OUT_DIR
    tab_dir = OUT_DIR
    supp_dir = OUT_DIR
    revision_dir = OUT_DIR
    hq_fig_dir = OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    preflight_plotting_check(OUT_DIR)

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {CSV_PATH}\n"
            "Run this program from the repository root so that the frozen "
            "cohort in data/ can be found."
        )

    df = load_and_filter_data(CSV_PATH)
    if len(df) == 0:
        print("ERROR: No data remaining after filters.")
        sys.exit(1)

    print("\nPreparing features...")
    le = LabelEncoder().fit(sorted(df["lineage"].unique()))
    y = le.transform(df["lineage"])
    X_clean = build_features(df["sequence"])
    print(f"  Feature matrix: {X_clean.shape}")
    print(f"  Classes: {le.classes_}")

    print("\n  Class distribution (imbalanced):")
    for i, lin in enumerate(le.classes_):
        count = np.sum(y == i)
        pct = count / len(y) * 100
        bar = "█" * (int(count / 20) if count < 500 else 40)
        print(f"    {lin:<14}  {count:>5}  {pct:>5.1f}%  {bar}")

    t0 = time.time()

    # The seed-42 analysis provides the reference results and the data behind
    # the figures. Nothing is written unless the switch above is enabled.
    print("\nRunning original seed-42 reference analysis in memory...")
    results_abc = run_conditions_abc(X_clean, y, le, df)
    results_d = run_condition_d(X_clean, y, le, df)

    if REGENERATE_ORIGINAL_OUTPUTS:
        print("\nREGENERATE_ORIGINAL_OUTPUTS=True: writing submitted-style outputs")
        save_results_abc(results_abc, tab_dir)
        save_results_d(results_d, tab_dir)
        generate_summary_tables(results_abc, results_d, le, df, X_clean, y, tab_dir)
        plot_conditions_abc(results_abc, fig_dir)
        plot_condition_d(results_d, fig_dir)
        plot_confusion_matrices_cv(X_clean, y, le, df, fig_dir)
        plot_per_lineage_f1_cv(X_clean, y, le, df, fig_dir)
        plot_feature_importance_supplementary(X_clean, y, supp_dir)
    else:
        print("\nSubmitted tables/figures were NOT overwritten.")

    if RUN_HQ_FIGURES:
        print("\n" + "="*70)
        print("GENERATING HIGH-QUALITY FIGURE COPIES")
        print("="*70)
        # Figures 1 and 2 come later, after the repeated statistics, so that
        # they can carry 95% CI error bars.
        plot_confusion_matrices_hq(X_clean, y, le, hq_fig_dir)
        plot_per_lineage_f1_hq(X_clean, y, le, df, hq_fig_dir)
        plot_feature_importance_hq(X_clean, y, hq_fig_dir)

    baseline_raw = None
    baseline_summary = None
    d_summary = None
    abc_summary = None
    e_summary = None
    if RUN_REPEATED_BASELINE:
        baseline_raw, baseline_summary, lineage_raw, lineage_summary = run_repeated_baseline(
            X_clean, y, le, original_results_abc=results_abc
        )
        baseline_raw.to_csv(os.path.join(revision_dir, 'Baseline_Repeated_Raw.csv'), index=False, encoding='utf-8-sig')
        baseline_summary.to_csv(os.path.join(revision_dir, 'Baseline_Repeated_Summary.csv'), index=False, encoding='utf-8-sig')
        lineage_raw.to_csv(os.path.join(revision_dir, 'Baseline_PerLineage_Recall_Repeated_Raw.csv'), index=False, encoding='utf-8-sig')
        lineage_summary.to_csv(os.path.join(revision_dir, 'Baseline_PerLineage_Recall_Repeated_Summary.csv'), index=False, encoding='utf-8-sig')
        paired = build_baseline_paired_comparison(baseline_raw)
        paired.to_csv(os.path.join(revision_dir, 'Baseline_XGBoost_vs_RF_Paired_Differences.csv'), index=False, encoding='utf-8-sig')
        print(f"\nBaseline repeated statistics saved to: {revision_dir}")

    if RUN_REPEATED_CONDITION_D:
        d_raw, d_summary, d_folds = run_repeated_condition_d(
            X_clean, y, le, original_results_d=results_d
        )
        d_raw.to_csv(os.path.join(revision_dir, 'ConditionD_Repeated_Raw.csv'), index=False, encoding='utf-8-sig')
        d_summary.to_csv(os.path.join(revision_dir, 'ConditionD_Repeated_Summary.csv'), index=False, encoding='utf-8-sig')
        d_folds.to_csv(os.path.join(revision_dir, 'ConditionD_Repeated_Fold_Details.csv'), index=False, encoding='utf-8-sig')
        print(f"Condition D repeated statistics saved to: {revision_dir}")

    if RUN_REPEATED_ABC:
        abc_raw, abc_summary = run_repeated_conditions_abc(
            X_clean, y, le, df, baseline_raw=baseline_raw, original_results_abc=results_abc
        )
        abc_raw.to_csv(os.path.join(revision_dir, 'ConditionsABC_Repeated_Raw.csv'), index=False, encoding='utf-8-sig')
        abc_summary.to_csv(os.path.join(revision_dir, 'ConditionsABC_Repeated_Summary.csv'), index=False, encoding='utf-8-sig')
        abc_paired = build_abc_paired_comparisons(abc_raw)
        abc_paired.to_csv(os.path.join(revision_dir, 'ConditionsABC_XGBoost_vs_RF_Paired_Differences.csv'),
                          index=False, encoding='utf-8-sig')
        print(f"Conditions A/B/C repeated statistics saved to: {revision_dir}")

    if RUN_REPEATED_CONDITION_E:
        e_raw, e_summary = run_repeated_condition_e(X_clean, y, le)
        e_raw.to_csv(os.path.join(revision_dir, 'ConditionE_Repeated_Raw.csv'), index=False, encoding='utf-8-sig')
        e_summary.to_csv(os.path.join(revision_dir, 'ConditionE_Repeated_Summary.csv'), index=False, encoding='utf-8-sig')
        print(f"Condition E repeated statistics saved to: {revision_dir}")
        if d_summary is not None and len(d_summary):
            de_comparison = build_condition_d_e_comparison(d_summary, e_summary)
            de_comparison.to_csv(os.path.join(revision_dir, 'ConditionD_vs_E_Burden_Comparison.csv'),
                                 index=False, encoding='utf-8-sig')

    if RUN_HQ_FIGURES:
        if abc_summary is not None and len(abc_summary):
            plot_conditions_abc_repeated_hq(abc_summary, hq_fig_dir)
        else:
            plot_conditions_abc_hq(results_abc, hq_fig_dir)
        if d_summary is not None and len(d_summary):
            plot_condition_d_repeated_hq(d_summary, hq_fig_dir)
        else:
            plot_condition_d_hq(results_d, hq_fig_dir)
        if e_summary is not None and len(e_summary):
            plot_condition_e_repeated_hq(e_summary, hq_fig_dir)

    write_revision_config(os.path.join(revision_dir, 'Revision_Statistics_Config.txt'), df, X_clean)

    manifest_path = os.path.join(OUT_DIR, 'README_Output_Folders.txt')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write('APBT revision pipeline outputs\n')
        f.write('=============================\n')
        f.write(f'Root folder: {OUT_DIR}\n\n')
        f.write('All CSV, TXT, PNG and PDF outputs are stored directly in this single folder.\n')
        f.write('The run contains the clean baseline, Conditions A-C, Condition D replacement, Condition E injection, repeated-CV statistics, and publication-quality figures.\n')

    with open(os.path.join(OUT_DIR, 'RUN_COMPLETE.txt'), 'w', encoding='utf-8') as f:
        f.write(f'APBT reviewer revision pipeline completed successfully at {datetime.now().isoformat(timespec="seconds")}\n')
        f.write('Reviewer comment 1: Condition D replacement + Condition E injection completed.\n')
        f.write('Reviewer comment 3: repeated 5-fold CV across seeds 42-51 completed for baseline and Conditions A-D; Condition E also repeated.\n')

    elapsed = time.time() - t0
    print("\n" + "="*70)
    print("REVISION RUN COMPLETE")
    print("="*70)
    print(f"Runtime: {elapsed/60:.1f} minutes")
    print(f"Original submitted outputs preserved: {not REGENERATE_ORIGINAL_OUTPUTS}")
    print(f"Revision statistics folder: {revision_dir}")
    print(f"Publication-quality figures folder: {hq_fig_dir}")
    print("\nKey files to share after the complete run:")
    print("  Baseline_Repeated_Summary.csv")
    print("  Baseline_XGBoost_vs_RF_Paired_Differences.csv")
    print("  ConditionsABC_Repeated_Summary.csv")
    print("  ConditionsABC_XGBoost_vs_RF_Paired_Differences.csv")
    print("  ConditionD_Repeated_Summary.csv")
    print("  ConditionE_Repeated_Summary.csv")
    print("  Fig1_Conditions_ABC_Degradation.png")
    print("  Fig2_Condition_D_Targeted_Attack.png")
    print("  Fig5_Condition_E_Targeted_Label_Injection.png")
    print("="*70)


if __name__ == "__main__":
    main()
