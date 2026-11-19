"""Matched controls for Condition D.

Separates directed targeted relabelling from three alternatives that use an
identical corruption budget:

    1) removal of correctly labelled source class training records,
    2) random incorrect relabelling of the same source records,
    3) targeted relabelling to every alternative destination class.

For a given repeat seed, fold, source lineage and rate, one deterministic
ordering of source lineage training records is generated. The same selected
records and the same absolute budget m are then reused by every condition, and
a checksum over the selection confirms it. Rates are 10, 20 and 30 per cent,
where correctly labelled source records remain available throughout.

Reads the frozen cohort from data/cohort.csv.xz. Writes all tables, figures
and run metadata into reproduced/matched_controls/. A checkpoint is written
after each completed experiment so an interrupted run resumes safely; set
FORCE_RESTART below to discard it.

Run from the repository root:

    python src/matched_controls.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SCRIPT_VERSION = "matched_conditionD_controls_v1"
DATASET_FILENAME = "cohort.csv.xz"

SEED_START = 42
N_REPEATS = 10
N_FOLDS = 5
N_TREES = 100
K = 4
T_CRIT_95_DF9 = 2.2621571627409915

# Correctly labelled source class records remain available at all three rates.
CONTROL_RATES = [0.10, 0.20, 0.30]

TARGET_LINEAGES = [
    "B.1.1.7", "P.1", "B.1.617.2", "BA.1", "BA.2", "BA.5", "BQ.1"
]

# The Condition D mappings, used as the primary targeted comparison.
PRIMARY_DESTINATION = {
    "BQ.1": "BA.5",
    "BA.5": "BA.2",
    "P.1": "B.1.617.2",
}

SOURCE_LINEAGES = list(PRIMARY_DESTINATION.keys())

# Set to True only if you want to discard an existing checkpoint and rerun all.
FORCE_RESTART = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / DATASET_FILENAME
OUT = Path("reproduced") / "matched_controls"
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT / "matched_controls_progress_checkpoint.csv"

if not DATA.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA}\n"
        "Run this program from the repository root so that the frozen "
        "cohort in data/ can be found."
    )


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

BASES = ["A", "C", "G", "T"]
NFEAT = 4 ** K
_CODE = np.full(256, -1, dtype=np.int16)
for _i, _b in enumerate(BASES):
    _CODE[ord(_b)] = _i


def to_vec(seq: str) -> np.ndarray:
    """Whole-genome normalised tetranucleotide-frequency vector."""
    s = str(seq).upper().replace("N", "A")
    a = _CODE[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]
    a = a[a >= 0]

    if len(a) < K:
        return np.zeros(NFEAT, dtype=np.float32)

    idx = np.zeros(len(a) - K + 1, dtype=np.int32)
    for j in range(K):
        idx = idx * 4 + a[j: len(a) - K + 1 + j]

    v = np.bincount(idx, minlength=NFEAT).astype(np.float32)
    total = v.sum()
    return v / total if total else v


def load_and_filter_data() -> pd.DataFrame:
    print(f"Loading dataset: {DATA}")
    df = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)

    required = {
        "accession", "lineage", "sequence", "sequence_length", "n_content_pct"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[df["lineage"].astype(str).isin(TARGET_LINEAGES)].copy()
    df = df[(df["sequence_length"] >= 29000) & (df["sequence_length"] <= 30200)].copy()
    df = df[df["n_content_pct"] <= 1.0].copy()
    df = df.drop_duplicates(subset=["accession", "sequence"]).reset_index(drop=True)

    print(f"Filtered dataset: {len(df):,} records")
    print(df["lineage"].value_counts().to_string())

    if len(df) != 6675:
        print(
            "WARNING: The filtered dataset does not contain 6,675 records. "
            "Check that you are using the frozen study dataset."
        )

    return df


def build_features(df: pd.DataFrame) -> np.ndarray:
    print("Building 256-dimensional whole-genome 4-mer features...")
    X = np.vstack([to_vec(s) for s in df["sequence"]])
    print(f"Feature matrix: {X.shape}")
    return X


def get_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=seed,
        n_jobs=-1,
    )


# -----------------------------------------------------------------------------
# Reproducible matched source-record selection
# -----------------------------------------------------------------------------


def stable_seed(*parts) -> int:
    """Stable 32-bit seed independent of Python's process-randomised hash()."""
    msg = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(msg).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def selection_plan(
    train_idx: np.ndarray,
    y: np.ndarray,
    source_class: int,
    source_name: str,
    repeat_seed: int,
    fold_number: int,
):
    """
    Create one deterministic source-record ordering for a repeat/fold/source.

    The same ordering is reused by every control and every targeted destination.
    Random incorrect labels are also generated once for the full ordering, so a
    selected record keeps the same random wrong label as the rate increases.
    """
    y_train = y[train_idx]
    source_positions = np.where(y_train == source_class)[0]

    rng_order = np.random.RandomState(
        stable_seed(SCRIPT_VERSION, repeat_seed, fold_number, source_name, "order")
    )
    ordered_positions = rng_order.permutation(source_positions)

    wrong_classes = np.array([c for c in np.unique(y) if c != source_class], dtype=int)
    rng_labels = np.random.RandomState(
        stable_seed(SCRIPT_VERSION, repeat_seed, fold_number, source_name, "random_labels")
    )
    random_wrong_labels = rng_labels.choice(
        wrong_classes, size=len(ordered_positions), replace=True
    )

    return ordered_positions, random_wrong_labels


def checksum_selected(train_idx: np.ndarray, selected_positions: np.ndarray) -> str:
    """Short checksum proving that matched conditions used the same records."""
    if len(selected_positions) == 0:
        return hashlib.sha256(b"").hexdigest()[:16]
    global_idx = train_idx[selected_positions]
    payload = ",".join(map(str, global_idx.tolist())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# -----------------------------------------------------------------------------
# Training-set modifications
# -----------------------------------------------------------------------------


def make_modified_training(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    selected_positions: np.ndarray,
    mode: str,
    random_wrong_labels: np.ndarray | None = None,
    destination_class: int | None = None,
):
    """Create one matched control/targeted training fold."""
    X_train = X[train_idx]
    y_train = y[train_idx].copy()

    if mode == "removal":
        keep = np.ones(len(train_idx), dtype=bool)
        keep[selected_positions] = False
        return X_train[keep], y_train[keep]

    if mode == "random_relabel":
        if random_wrong_labels is None:
            raise ValueError("random_wrong_labels required for random_relabel")
        y_train[selected_positions] = random_wrong_labels
        return X_train, y_train

    if mode == "targeted":
        if destination_class is None:
            raise ValueError("destination_class required for targeted mode")
        y_train[selected_positions] = destination_class
        return X_train, y_train

    raise ValueError(f"Unknown mode: {mode}")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def source_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source_class: int,
    primary_destination_class: int,
    selected_destination_class: int | None,
):
    acc = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    src_recall = recall_score(
        y_true,
        y_pred,
        labels=[source_class],
        average="macro",
        zero_division=0,
    ) * 100.0

    mask = y_true == source_class
    if mask.sum() == 0:
        primary_dest_rate = np.nan
        selected_dest_rate = np.nan
    else:
        primary_dest_rate = float(np.mean(y_pred[mask] == primary_destination_class) * 100.0)
        if selected_destination_class is None:
            selected_dest_rate = np.nan
        else:
            selected_dest_rate = float(np.mean(y_pred[mask] == selected_destination_class) * 100.0)

    return acc, macro_f1, src_recall, primary_dest_rate, selected_dest_rate


def mean_sd_ci95(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, 0.0, mean, mean
    sd = float(np.std(arr, ddof=1))
    if len(arr) == 10:
        tcrit = T_CRIT_95_DF9
    else:
        # Conservative normal approximation fallback; should not occur in the
        # intended full 10-repeat analysis.
        tcrit = 1.96
    half = tcrit * sd / np.sqrt(len(arr))
    return mean, sd, mean - half, mean + half


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------


def load_checkpoint() -> pd.DataFrame:
    if FORCE_RESTART and CHECKPOINT.exists():
        CHECKPOINT.unlink()

    if not CHECKPOINT.exists():
        return pd.DataFrame()

    cp = pd.read_csv(CHECKPOINT)
    if "script_version" not in cp.columns or not (cp["script_version"] == SCRIPT_VERSION).all():
        raise RuntimeError(
            f"Existing checkpoint {CHECKPOINT} is from a different script version. "
            "Rename/delete it or set FORCE_RESTART=True."
        )
    print(f"Resuming from checkpoint with {len(cp):,} completed repeat-level experiments.")
    return cp


def experiment_key(row_or_dict) -> tuple:
    def get(k):
        return row_or_dict[k] if isinstance(row_or_dict, dict) else getattr(row_or_dict, k)

    destination = get("destination_lineage")
    if pd.isna(destination):
        destination = ""
    return (
        int(get("repeat_seed")),
        str(get("source_lineage")),
        float(get("rate_pct")),
        str(get("condition")),
        str(destination),
    )


def append_checkpoint(row: dict):
    frame = pd.DataFrame([row])
    header = not CHECKPOINT.exists()
    frame.to_csv(CHECKPOINT, mode="a", header=header, index=False)


# -----------------------------------------------------------------------------
# One repeat-level experiment pooled across five clean test folds
# -----------------------------------------------------------------------------


def run_one_experiment(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    le: LabelEncoder,
    repeat_seed: int,
    source_name: str,
    rate: float,
    condition: str,
    destination_name: str | None,
):
    source_class = int(le.transform([source_name])[0])
    primary_destination_name = PRIMARY_DESTINATION[source_name]
    primary_destination_class = int(le.transform([primary_destination_name])[0])
    destination_class = (
        int(le.transform([destination_name])[0]) if destination_name is not None else None
    )

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=repeat_seed,
    )

    y_true_oof = []
    y_pred_oof = []
    m_values = []
    n_source_values = []
    remaining_values = []
    train_size_before_values = []
    train_size_after_values = []
    fold_checksums = []

    for fold_number, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        ordered_positions, wrong_labels_all = selection_plan(
            train_idx=train_idx,
            y=y,
            source_class=source_class,
            source_name=source_name,
            repeat_seed=repeat_seed,
            fold_number=fold_number,
        )

        n_source = len(ordered_positions)
        m = int(round(n_source * rate))
        if rate > 0 and m == 0:
            m = 1
        m = min(m, n_source)

        selected_positions = ordered_positions[:m]
        random_wrong_labels = wrong_labels_all[:m]
        fold_checksums.append(checksum_selected(train_idx, selected_positions))

        if condition == "removal":
            X_mod, y_mod = make_modified_training(
                X, y, train_idx, selected_positions, mode="removal"
            )
        elif condition == "random_relabel":
            X_mod, y_mod = make_modified_training(
                X,
                y,
                train_idx,
                selected_positions,
                mode="random_relabel",
                random_wrong_labels=random_wrong_labels,
            )
        elif condition == "targeted":
            X_mod, y_mod = make_modified_training(
                X,
                y,
                train_idx,
                selected_positions,
                mode="targeted",
                destination_class=destination_class,
            )
        else:
            raise ValueError(condition)

        clf = get_rf(repeat_seed)
        sample_weights = compute_sample_weight("balanced", y_mod)
        clf.fit(X_mod, y_mod, sample_weight=sample_weights)
        pred = clf.predict(X[test_idx])

        y_true_oof.extend(y[test_idx].tolist())
        y_pred_oof.extend(pred.tolist())

        m_values.append(m)
        n_source_values.append(n_source)
        remaining_values.append(n_source - m)
        train_size_before_values.append(len(train_idx))
        train_size_after_values.append(len(y_mod))

    y_true_oof = np.asarray(y_true_oof, dtype=int)
    y_pred_oof = np.asarray(y_pred_oof, dtype=int)

    acc, macro_f1, src_recall, primary_dest_rate, selected_dest_rate = source_metrics(
        y_true_oof,
        y_pred_oof,
        source_class,
        primary_destination_class,
        destination_class if condition == "targeted" else None,
    )

    # One checksum over the five fold-specific checksums. All matched conditions
    # for the same repeat/source/rate must have exactly the same value.
    selection_checksum = hashlib.sha256(
        "|".join(fold_checksums).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "script_version": SCRIPT_VERSION,
        "repeat_seed": repeat_seed,
        "source_lineage": source_name,
        "primary_destination": primary_destination_name,
        "rate_pct": rate * 100.0,
        "condition": condition,
        "destination_lineage": destination_name if destination_name is not None else "",
        "is_primary_destination": bool(
            condition == "targeted" and destination_name == primary_destination_name
        ),
        "mean_source_records_in_training_fold": float(np.mean(n_source_values)),
        "mean_selected_records_m": float(np.mean(m_values)),
        "mean_correct_source_records_remaining": float(np.mean(remaining_values)),
        "mean_original_training_fold_size": float(np.mean(train_size_before_values)),
        "mean_modified_training_fold_size": float(np.mean(train_size_after_values)),
        "mean_full_fold_budget_pct": float(
            np.mean(np.asarray(m_values) / np.asarray(train_size_before_values)) * 100.0
        ),
        "selection_checksum": selection_checksum,
        "accuracy_pct": acc,
        "macro_f1_pct": macro_f1,
        "source_recall_pct": src_recall,
        "primary_destination_prediction_rate_pct": primary_dest_rate,
        "selected_destination_prediction_rate_pct": selected_dest_rate,
    }


# -----------------------------------------------------------------------------
# Clean baseline for each repeat (one RF per fold, reused across source metrics)
# -----------------------------------------------------------------------------


def run_clean_baselines(X, y, le):
    rows = []
    print("\nRunning clean repeated-CV baseline...")

    for repeat_seed in range(SEED_START, SEED_START + N_REPEATS):
        cv = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=repeat_seed,
        )
        y_true_oof = []
        y_pred_oof = []

        for train_idx, test_idx in cv.split(X, y):
            y_train = y[train_idx]
            clf = get_rf(repeat_seed)
            weights = compute_sample_weight("balanced", y_train)
            clf.fit(X[train_idx], y_train, sample_weight=weights)
            pred = clf.predict(X[test_idx])
            y_true_oof.extend(y[test_idx].tolist())
            y_pred_oof.extend(pred.tolist())

        y_true_oof = np.asarray(y_true_oof, dtype=int)
        y_pred_oof = np.asarray(y_pred_oof, dtype=int)

        for source_name in SOURCE_LINEAGES:
            source_class = int(le.transform([source_name])[0])
            primary_name = PRIMARY_DESTINATION[source_name]
            primary_class = int(le.transform([primary_name])[0])
            acc, mf1, rec, primary_rate, _ = source_metrics(
                y_true_oof, y_pred_oof, source_class, primary_class, None
            )
            rows.append({
                "script_version": SCRIPT_VERSION,
                "repeat_seed": repeat_seed,
                "source_lineage": source_name,
                "primary_destination": primary_name,
                "rate_pct": 0.0,
                "condition": "clean",
                "destination_lineage": "",
                "is_primary_destination": False,
                "mean_source_records_in_training_fold": np.nan,
                "mean_selected_records_m": 0.0,
                "mean_correct_source_records_remaining": np.nan,
                "mean_original_training_fold_size": np.nan,
                "mean_modified_training_fold_size": np.nan,
                "mean_full_fold_budget_pct": 0.0,
                "selection_checksum": "",
                "accuracy_pct": acc,
                "macro_f1_pct": mf1,
                "source_recall_pct": rec,
                "primary_destination_prediction_rate_pct": primary_rate,
                "selected_destination_prediction_rate_pct": np.nan,
            })

        print(f"  seed {repeat_seed}: accuracy={acc:.2f}%")

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def summarise_repeat_results(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "source_lineage",
        "primary_destination",
        "rate_pct",
        "condition",
        "destination_lineage",
        "is_primary_destination",
    ]

    metrics = [
        "accuracy_pct",
        "macro_f1_pct",
        "source_recall_pct",
        "primary_destination_prediction_rate_pct",
        "selected_destination_prediction_rate_pct",
    ]

    rows = []
    for keys, g in raw.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_repeats"] = int(g["repeat_seed"].nunique())

        for col in [
            "mean_source_records_in_training_fold",
            "mean_selected_records_m",
            "mean_correct_source_records_remaining",
            "mean_original_training_fold_size",
            "mean_modified_training_fold_size",
            "mean_full_fold_budget_pct",
        ]:
            row[col] = float(g[col].mean()) if g[col].notna().any() else np.nan

        for metric in metrics:
            mean, sd, lo, hi = mean_sd_ci95(g[metric].values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi

        rows.append(row)

    return pd.DataFrame(rows)


def make_primary_paired_comparisons(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Paired repeat-level differences for the original Condition D destination
    versus removal and random incorrect relabelling.

    Difference is TARGETED PRIMARY minus CONTROL. Negative source-recall values
    therefore mean that targeted relabelling produced lower recall than control.
    """
    rows = []
    metrics = [
        "source_recall_pct",
        "accuracy_pct",
        "macro_f1_pct",
        "primary_destination_prediction_rate_pct",
    ]

    targeted = raw[(raw["condition"] == "targeted") & (raw["is_primary_destination"] == True)].copy()

    for source in SOURCE_LINEAGES:
        for rate in [r * 100.0 for r in CONTROL_RATES]:
            t = targeted[
                (targeted["source_lineage"] == source) &
                (np.isclose(targeted["rate_pct"], rate))
            ]

            for control_name in ["removal", "random_relabel"]:
                c = raw[
                    (raw["condition"] == control_name) &
                    (raw["source_lineage"] == source) &
                    (np.isclose(raw["rate_pct"], rate))
                ]

                merged = t.merge(
                    c,
                    on=["repeat_seed", "source_lineage", "rate_pct"],
                    suffixes=("_targeted", "_control"),
                )

                if len(merged) != N_REPEATS:
                    print(
                        f"WARNING: paired comparison {source} {rate}% {control_name} "
                        f"has {len(merged)} repeats, expected {N_REPEATS}."
                    )

                base = {
                    "source_lineage": source,
                    "primary_destination": PRIMARY_DESTINATION[source],
                    "rate_pct": rate,
                    "control": control_name,
                    "n_paired_repeats": len(merged),
                }

                for metric in metrics:
                    diffs = (
                        merged[f"{metric}_targeted"].values -
                        merged[f"{metric}_control"].values
                    )
                    mean, sd, lo, hi = mean_sd_ci95(diffs)
                    base[f"diff_{metric}_mean"] = mean
                    base[f"diff_{metric}_sd"] = sd
                    base[f"diff_{metric}_ci95_low"] = lo
                    base[f"diff_{metric}_ci95_high"] = hi

                rows.append(base)

    return pd.DataFrame(rows)


def make_destination_range(summary: pd.DataFrame) -> pd.DataFrame:
    t = summary[summary["condition"] == "targeted"].copy()
    rows = []

    for (source, rate), g in t.groupby(["source_lineage", "rate_pct"], sort=False):
        recall_col = "source_recall_pct_mean"
        min_row = g.loc[g[recall_col].idxmin()]
        max_row = g.loc[g[recall_col].idxmax()]
        primary = g[g["is_primary_destination"] == True]

        rows.append({
            "source_lineage": source,
            "rate_pct": rate,
            "n_destinations_tested": len(g),
            "minimum_source_recall_pct": float(min_row[recall_col]),
            "destination_with_minimum_recall": min_row["destination_lineage"],
            "maximum_source_recall_pct": float(max_row[recall_col]),
            "destination_with_maximum_recall": max_row["destination_lineage"],
            "destination_range_width_pct_points": float(
                max_row[recall_col] - min_row[recall_col]
            ),
            "primary_destination": PRIMARY_DESTINATION[source],
            "primary_destination_source_recall_pct": (
                float(primary.iloc[0][recall_col]) if len(primary) else np.nan
            ),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------


def safe_name(lineage: str) -> str:
    return lineage.replace(".", "_").replace("/", "_")


def make_figures(summary: pd.DataFrame):
    rates = [r * 100.0 for r in CONTROL_RATES]

    for source in SOURCE_LINEAGES:
        # Figure 1 per source: primary targeted vs matched controls.
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        series = [
            ("removal", None, "Removal control"),
            ("random_relabel", None, "Random incorrect relabelling"),
            ("targeted", PRIMARY_DESTINATION[source], f"Targeted to {PRIMARY_DESTINATION[source]}"),
        ]

        for condition, destination, label in series:
            g = summary[
                (summary["source_lineage"] == source) &
                (summary["condition"] == condition)
            ].copy()
            if destination is not None:
                g = g[g["destination_lineage"] == destination]
            g = g.sort_values("rate_pct")
            ax.errorbar(
                g["rate_pct"],
                g["source_recall_pct_mean"],
                yerr=[
                    g["source_recall_pct_mean"] - g["source_recall_pct_ci95_low"],
                    g["source_recall_pct_ci95_high"] - g["source_recall_pct_mean"],
                ],
                marker="o",
                capsize=3,
                label=label,
            )

        ax.set_xlabel("Matched source-class corruption rate (%)")
        ax.set_ylabel(f"{source} recall (%)")
        ax.set_title(f"Matched controls for {source}")
        ax.set_xticks(rates)
        ax.set_ylim(0, 105)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            OUT / f"Fig_MatchedControls_{safe_name(source)}_PrimaryComparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Figure 2 per source: sensitivity to destination choice.
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        gsource = summary[
            (summary["source_lineage"] == source) &
            (summary["condition"] == "targeted")
        ].copy()

        for destination in TARGET_LINEAGES:
            if destination == source:
                continue
            g = gsource[gsource["destination_lineage"] == destination].sort_values("rate_pct")
            if g.empty:
                continue
            label = destination
            if destination == PRIMARY_DESTINATION[source]:
                label += " (original D)"
            ax.plot(
                g["rate_pct"],
                g["source_recall_pct_mean"],
                marker="o",
                label=label,
            )

        ax.set_xlabel("Matched source-class corruption rate (%)")
        ax.set_ylabel(f"{source} recall (%)")
        ax.set_title(f"Destination sensitivity for {source}")
        ax.set_xticks(rates)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            OUT / f"Fig_MatchedControls_{safe_name(source)}_DestinationSensitivity.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


# -----------------------------------------------------------------------------
# Validation checks
# -----------------------------------------------------------------------------


def validate_matching(raw: pd.DataFrame):
    """Fail loudly if supposedly matched conditions used different records."""
    controls = raw[raw["condition"].isin(["removal", "random_relabel", "targeted"])].copy()

    bad = []
    for keys, g in controls.groupby(["repeat_seed", "source_lineage", "rate_pct"], sort=False):
        checksums = set(g["selection_checksum"].dropna().astype(str))
        mvals = np.round(g["mean_selected_records_m"].values, 8)
        if len(checksums) != 1 or len(np.unique(mvals)) != 1:
            bad.append((keys, checksums, np.unique(mvals).tolist()))

    if bad:
        raise AssertionError(
            "Matched-control validation failed. Some conditions did not use "
            f"identical selected records/budgets. First problem: {bad[0]}"
        )

    print("Matched-control validation PASSED: same records and budgets across controls/destinations.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    print("=" * 78)
    print("MATCHED CONTROL ANALYSIS FOR CONDITION D")
    print("Removal vs random incorrect relabelling vs targeted destinations")
    print("=" * 78)
    print(f"Dataset: {DATA}")
    print(f"Outputs: {OUT}")
    print(f"Seeds: {SEED_START}-{SEED_START + N_REPEATS - 1}")
    print(f"Rates: {[int(r*100) for r in CONTROL_RATES]}%")
    print("Targeted alternative destinations: every other lineage")

    # Number of RF fits: clean baseline + all matched controls.
    n_target_destinations = len(TARGET_LINEAGES) - 1
    fits = (
        N_REPEATS * N_FOLDS +
        len(SOURCE_LINEAGES) * len(CONTROL_RATES) * N_REPEATS * N_FOLDS *
        (2 + n_target_destinations)
    )
    print(f"Planned Random Forest fits: {fits:,}")
    print("The run is checkpointed after every completed repeat-level experiment.")

    df = load_and_filter_data()
    X = build_features(df)
    le = LabelEncoder().fit(sorted(df["lineage"].astype(str).unique()))
    y = le.transform(df["lineage"].astype(str))

    print(f"Encoded classes: {list(le.classes_)}")

    # The clean baseline is cheap and always recalculated, so the output file
    # stands on its own and is directly comparable with the controls.
    baseline = run_clean_baselines(X, y, le)

    checkpoint_df = load_checkpoint()
    completed = set()
    if not checkpoint_df.empty:
        for row in checkpoint_df.itertuples(index=False):
            completed.add(experiment_key(row))

    total_experiments = (
        len(SOURCE_LINEAGES) * len(CONTROL_RATES) * N_REPEATS *
        (2 + (len(TARGET_LINEAGES) - 1))
    )
    done_counter = len(completed)

    for repeat_seed in range(SEED_START, SEED_START + N_REPEATS):
        for source in SOURCE_LINEAGES:
            destinations = [d for d in TARGET_LINEAGES if d != source]

            for rate in CONTROL_RATES:
                experiments = [
                    ("removal", None),
                    ("random_relabel", None),
                ] + [("targeted", d) for d in destinations]

                for condition, destination in experiments:
                    key = (
                        repeat_seed,
                        source,
                        rate * 100.0,
                        condition,
                        destination or "",
                    )
                    if key in completed:
                        continue

                    done_counter += 1
                    dest_txt = f" -> {destination}" if destination else ""
                    print(
                        f"[{done_counter:>4}/{total_experiments}] "
                        f"seed={repeat_seed} source={source} rate={rate*100:.0f}% "
                        f"{condition}{dest_txt}"
                    )

                    row = run_one_experiment(
                        X=X,
                        y=y,
                        df=df,
                        le=le,
                        repeat_seed=repeat_seed,
                        source_name=source,
                        rate=rate,
                        condition=condition,
                        destination_name=destination,
                    )
                    append_checkpoint(row)
                    completed.add(key)

    # Read the complete checkpoint after the run. This also makes resumed runs
    # identical to uninterrupted runs.
    controls = pd.read_csv(CHECKPOINT)
    raw = pd.concat([baseline, controls], ignore_index=True, sort=False)

    validate_matching(raw)

    summary = summarise_repeat_results(raw)
    paired = make_primary_paired_comparisons(raw)
    destination_range = make_destination_range(summary)

    raw_path = OUT / "matched_controls_repeat_level.csv"
    summary_path = OUT / "matched_controls_summary.csv"
    paired_path = OUT / "matched_controls_primary_paired_comparisons.csv"
    range_path = OUT / "matched_controls_destination_range.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    destination_range.to_csv(range_path, index=False)

    make_figures(summary)

    metadata = {
        "script_version": SCRIPT_VERSION,
        "dataset": DATASET_FILENAME,
        "n_records_after_filtering": int(len(df)),
        "lineage_counts": {
            k: int(v) for k, v in df["lineage"].value_counts().to_dict().items()
        },
        "feature_dimension": int(X.shape[1]),
        "kmer_size": K,
        "classifier": "RandomForestClassifier",
        "n_estimators": N_TREES,
        "balanced_sample_weights_recomputed_after_modification": True,
        "n_folds": N_FOLDS,
        "repeat_seeds": list(range(SEED_START, SEED_START + N_REPEATS)),
        "control_rates_pct": [r * 100.0 for r in CONTROL_RATES],
        "source_lineages": SOURCE_LINEAGES,
        "primary_destinations": PRIMARY_DESTINATION,
        "alternative_destination_rule": "every other lineage class",
        "matching_rule": (
            "Within each repeat/fold/source/rate, removal, random relabelling, "
            "and every targeted destination use the exact same selected source "
            "records and absolute budget m. Selection is nested by rate."
        ),
        "random_relabel_rule": (
            "Selected source records receive deterministic random labels drawn "
            "from all six non-source classes; assignments are nested/consistent "
            "as the rate increases within a repeat/fold/source."
        ),
        "ci_rule": (
            "Repeat-level metrics pool all five held-out folds; mean, sample SD, "
            "and two-sided 95% Student-t CI are calculated across 10 repeats."
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "outputs": [
            raw_path.name,
            summary_path.name,
            paired_path.name,
            range_path.name,
        ],
    }

    metadata_path = OUT / "matched_controls_run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 78)
    print("MATCHED CONTROL ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"Repeat-level results: {raw_path}")
    print(f"Summary:              {summary_path}")
    print(f"Paired comparisons:   {paired_path}")
    print(f"Destination ranges:   {range_path}")
    print(f"Metadata:              {metadata_path}")
    print(f"Figures:               {OUT}")


if __name__ == "__main__":
    main()
