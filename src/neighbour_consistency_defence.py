"""Neighbour consistency defence against targeted label manipulation.

Flags a training record when its nearest eligible training neighbours in
feature space support a different lineage label. The primary rule requires
three of five neighbours to disagree, the sensitivity rule four of five.
Detection sensitivity, flag precision and the false positive rate are
reported for Condition D and Condition E. Injected candidates are never used
as reference neighbours and the held-out folds take no part in the defence.

Reads the frozen cohort from data/cohort.csv.xz. Writes its tables, figures
and metadata into reproduced/defence_neighbour_consistency/.

Run from the repository root:

    python src/neighbour_consistency_defence.py
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


DATASET_NAME = "cohort.csv.xz"
REPEAT_SEEDS = list(range(42, 52))
N_FOLDS = 5
KMER_K = 4
N_NEIGHBOURS = 5
PRIMARY_MIN_CONSENSUS = 3
SENSITIVITY_MIN_CONSENSUS = 4
TARGETED_RATES = [0.0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
ATTACK_PAIRS = [
    ("BQ.1", "BA.5"),
    ("BA.5", "BA.2"),
    ("P.1", "B.1.617.2"),
]
PRECOMPUTED_NEIGHBOURS = 101
OUTDIR_NAME = "defence_neighbour_consistency"

BASES = ["A", "C", "G", "T"]
_CODE = np.full(256, -1, dtype=np.int16)
for _i, _b in enumerate(BASES):
    _CODE[ord(_b)] = _i


def find_dataset() -> Path:
    path = Path(__file__).resolve().parents[1] / "data" / DATASET_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run this program from the repository root so that the frozen "
            "cohort in data/ can be found."
        )
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_vec(seq: str) -> np.ndarray:
    s = str(seq).upper().replace("N", "A")
    a = _CODE[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]
    a = a[a >= 0]
    if len(a) < KMER_K:
        return np.zeros(4 ** KMER_K, dtype=np.float32)

    idx = np.zeros(len(a) - KMER_K + 1, dtype=np.int32)
    for j in range(KMER_K):
        idx = idx * 4 + a[j: len(a) - KMER_K + 1 + j]

    v = np.bincount(idx, minlength=4 ** KMER_K).astype(np.float32)
    total = v.sum()
    return v / total if total > 0 else v


def build_features(sequences) -> np.ndarray:
    return np.vstack([to_vec(s) for s in sequences])


def summary_stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return dict(mean=np.nan, sd=np.nan, ci95_low=np.nan, ci95_high=np.nan)
    mean = float(np.mean(a))
    if len(a) == 1:
        return dict(mean=mean, sd=np.nan, ci95_low=np.nan, ci95_high=np.nan)
    sd = float(np.std(a, ddof=1))
    tcrit = 2.2621571628540993 if len(a) == 10 else 1.96
    half = tcrit * sd / math.sqrt(len(a))
    return dict(mean=mean, sd=sd, ci95_low=mean-half, ci95_high=mean+half)


def vote_prediction(neighbour_matrix, labels, n_classes):
    labs = labels[neighbour_matrix]
    counts = np.stack(
        [(labs == c).sum(axis=1) for c in range(n_classes)],
        axis=1,
    )
    return counts.argmax(axis=1), counts.max(axis=1)


def build_fold_neighbours(tr, global_neighbours, n_records, k):
    train_mask = np.zeros(n_records, dtype=bool)
    train_mask[tr] = True
    row_position = np.full(n_records, -1, dtype=int)
    row_position[tr] = np.arange(len(tr))

    exclude_self = np.empty((len(tr), k), dtype=int)
    include_self = np.empty((len(tr), k), dtype=int)

    for pos, i in enumerate(tr):
        a = []
        b = []
        for j in global_neighbours[i]:
            j = int(j)
            if not train_mask[j]:
                continue

            if len(b) < k:
                b.append(j)

            if j != i and len(a) < k:
                a.append(j)

            if len(a) == k and len(b) == k:
                break

        if len(a) < k or len(b) < k:
            raise RuntimeError("Insufficient training neighbours in precomputed list.")

        exclude_self[pos] = a
        include_self[pos] = b

    return exclude_self, include_self, row_position


def prepare_fold_cache(X, y, global_neighbours, min_consensus):
    n_records = len(y)
    n_classes = len(np.unique(y))
    cache = {}

    for seed in REPEAT_SEEDS:
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)

        for fold_number, (tr, te) in enumerate(cv.split(X, y), start=1):
            tr = np.asarray(tr)
            exclude_self, include_self, row_position = build_fold_neighbours(
                tr, global_neighbours, n_records, N_NEIGHBOURS
            )

            clean_pred, clean_votes = vote_prediction(
                exclude_self, y, n_classes
            )
            clean_flags = (
                (clean_pred != y[tr]) & (clean_votes >= min_consensus)
            )

            include_pred, include_votes = vote_prediction(
                include_self, y, n_classes
            )

            cache[(seed, fold_number)] = {
                "tr": tr,
                "exclude_self": exclude_self,
                "include_self": include_self,
                "row_position": row_position,
                "clean_flags": clean_flags,
                "include_pred": include_pred,
                "include_votes": include_votes,
            }

    return cache


def run_rule(X, y, le, global_neighbours, min_consensus, rule_name):
    n_classes = len(le.classes_)
    cache = prepare_fold_cache(X, y, global_neighbours, min_consensus)
    rows = []

    for condition in ["D", "E"]:
        for source_name, destination_name in ATTACK_PAIRS:
            source_class = int(le.transform([source_name])[0])
            destination_class = int(le.transform([destination_name])[0])

            for rate in TARGETED_RATES:
                for repeat_number, seed in enumerate(REPEAT_SEEDS, start=1):
                    TP = FP = M = LEGIT = FLAGS = 0

                    for fold_number in range(1, N_FOLDS + 1):
                        item = cache[(seed, fold_number)]
                        tr = item["tr"]

                        source_positions = np.where(y[tr] == source_class)[0]
                        n_source = len(source_positions)
                        m = 0 if rate == 0 else max(
                            1, int(round(n_source * rate))
                        )

                        if condition == "D":
                            current_labels = y.copy()
                            manipulated_global = np.array([], dtype=int)

                            if m > 0:
                                rng = np.random.RandomState(
                                    seed + fold_number * 1009
                                )
                                selected_positions = rng.permutation(
                                    source_positions
                                )[:m]
                                manipulated_global = tr[selected_positions]
                                current_labels[manipulated_global] = destination_class

                            pred, votes = vote_prediction(
                                item["exclude_self"],
                                current_labels,
                                n_classes,
                            )
                            flags = (
                                (pred != current_labels[tr])
                                & (votes >= min_consensus)
                            )
                            is_manipulated = np.isin(
                                tr, manipulated_global
                            )

                            tp = int(np.sum(flags & is_manipulated))
                            fp = int(np.sum(flags & ~is_manipulated))

                            TP += tp
                            FP += fp
                            M += m
                            LEGIT += len(tr) - m
                            FLAGS += int(np.sum(flags))

                        else:
                            if m > 0:
                                rng = np.random.RandomState(seed)
                                selected_positions = rng.choice(
                                    source_positions,
                                    size=m,
                                    replace=False,
                                )
                                injected_source_indices = tr[selected_positions]
                                local_positions = item["row_position"][
                                    injected_source_indices
                                ]

                                injected_flags = (
                                    (
                                        item["include_pred"][local_positions]
                                        != destination_class
                                    )
                                    & (
                                        item["include_votes"][local_positions]
                                        >= min_consensus
                                    )
                                )
                            else:
                                injected_flags = np.array([], dtype=bool)

                            tp = int(np.sum(injected_flags))
                            fp = int(np.sum(item["clean_flags"]))

                            TP += tp
                            FP += fp
                            M += m
                            LEGIT += len(tr)
                            FLAGS += tp + fp

                    sensitivity = (
                        100.0 * TP / M if M > 0 else np.nan
                    )
                    precision = (
                        100.0 * TP / FLAGS if FLAGS > 0 else 0.0
                    )
                    fpr = (
                        100.0 * FP / LEGIT if LEGIT > 0 else np.nan
                    )

                    rows.append({
                        "rule": rule_name,
                        "condition": condition,
                        "source_lineage": source_name,
                        "destination_lineage": destination_name,
                        "source_rate_pct": rate * 100.0,
                        "repeat": repeat_number,
                        "repeat_seed": seed,
                        "true_positive_flags": TP,
                        "false_positive_flags": FP,
                        "manipulated_record_instances": M,
                        "legitimate_record_instances": LEGIT,
                        "all_flags": FLAGS,
                        "detection_sensitivity_pct": sensitivity,
                        "flag_precision_pct": precision,
                        "false_positive_rate_pct": fpr,
                    })

    return pd.DataFrame(rows)


def summarise(raw):
    rows = []
    keys = [
        "rule", "condition", "source_lineage",
        "destination_lineage", "source_rate_pct",
    ]

    for key, g in raw.groupby(keys, sort=False):
        sens = summary_stats(g["detection_sensitivity_pct"])
        prec = summary_stats(g["flag_precision_pct"])
        fpr = summary_stats(g["false_positive_rate_pct"])

        rows.append({
            "rule": key[0],
            "condition": key[1],
            "source_lineage": key[2],
            "destination_lineage": key[3],
            "source_rate_pct": key[4],
            "n_repeats": len(g),
            "detection_sensitivity_mean_pct": sens["mean"],
            "detection_sensitivity_sd_pct": sens["sd"],
            "detection_sensitivity_ci95_low_pct": sens["ci95_low"],
            "detection_sensitivity_ci95_high_pct": sens["ci95_high"],
            "flag_precision_mean_pct": prec["mean"],
            "flag_precision_sd_pct": prec["sd"],
            "flag_precision_ci95_low_pct": prec["ci95_low"],
            "flag_precision_ci95_high_pct": prec["ci95_high"],
            "false_positive_rate_mean_pct": fpr["mean"],
            "false_positive_rate_sd_pct": fpr["sd"],
            "false_positive_rate_ci95_low_pct": fpr["ci95_low"],
            "false_positive_rate_ci95_high_pct": fpr["ci95_high"],
        })

    return pd.DataFrame(rows)


def write_tables(primary_summary, outdir):
    e = primary_summary[
        (primary_summary["condition"] == "E")
        & (primary_summary["source_rate_pct"].isin([30.0, 70.0, 100.0]))
    ].copy()

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Neighbour-consistency defence under Condition E. Values are means across 10 repeated five-fold analyses.}",
        r"\label{tab:conditionE_defence}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Source} & \textbf{Destination} & \textbf{Rate (\%)} & \textbf{Detection sensitivity (\%)} & \textbf{FPR (\%)} \\",
        r"\midrule",
    ]

    for _, row in e.iterrows():
        lines.append(
            f"{row['source_lineage']} & {row['destination_lineage']} & "
            f"{row['source_rate_pct']:.0f} & "
            f"{row['detection_sensitivity_mean_pct']:.2f} & "
            f"{row['false_positive_rate_mean_pct']:.2f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (outdir / "ConditionE_Defence_Main_Table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def make_figures(primary_summary, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for condition in ["D", "E"]:
        data = primary_summary[
            (primary_summary["condition"] == condition)
            & (primary_summary["source_rate_pct"] > 0)
        ]

        fig = plt.figure(figsize=(8, 5.2))
        ax = fig.add_subplot(111)

        for (src, dst), g in data.groupby(
            ["source_lineage", "destination_lineage"], sort=False
        ):
            g = g.sort_values("source_rate_pct")
            ax.plot(
                g["source_rate_pct"],
                g["detection_sensitivity_mean_pct"],
                marker="o",
                label=f"{src} → {dst}",
            )

        ax.set_xlabel("Source-relative manipulation rate (%)")
        ax.set_ylabel("Detection sensitivity (%)")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.set_title(
            f"Neighbour-consistency defence: Condition {condition}"
        )
        fig.tight_layout()

        fig.savefig(
            outdir / f"Fig_Defence_Condition_{condition}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            outdir / f"Fig_Defence_Condition_{condition}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def main():
    dataset = find_dataset()
    outdir = Path("reproduced") / OUTDIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset)

    required = {"accession", "lineage", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print(f"Dataset: {dataset.name}")
    print(f"Records: {len(df)}")

    X = build_features(df["sequence"])
    le = LabelEncoder()
    y = le.fit_transform(df["lineage"].astype(str))

    nn = NearestNeighbors(
        n_neighbors=min(PRECOMPUTED_NEIGHBOURS, len(df)),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(X)
    _, global_neighbours = nn.kneighbors(X)

    primary_raw = run_rule(
        X, y, le, global_neighbours,
        PRIMARY_MIN_CONSENSUS,
        "5NN_majority_3of5",
    )
    sensitivity_raw = run_rule(
        X, y, le, global_neighbours,
        SENSITIVITY_MIN_CONSENSUS,
        "5NN_supermajority_4of5",
    )

    primary_summary = summarise(primary_raw)
    sensitivity_summary = summarise(sensitivity_raw)

    raw = pd.concat(
        [primary_raw, sensitivity_raw], ignore_index=True
    )
    summary = pd.concat(
        [primary_summary, sensitivity_summary], ignore_index=True
    )

    raw.to_csv(
        outdir / "defence_raw_repeat_metrics.csv", index=False
    )
    summary.to_csv(
        outdir / "defence_summary_metrics.csv", index=False
    )

    write_tables(primary_summary, outdir)
    make_figures(primary_summary, outdir)

    metadata = {
        "dataset": dataset.name,
        "dataset_sha256": sha256_file(dataset),
        "n_records": int(len(df)),
        "feature_dimension": int(X.shape[1]),
        "lineages": list(le.classes_),
        "repeat_seeds": REPEAT_SEEDS,
        "n_folds": N_FOLDS,
        "distance_metric": "cosine",
        "primary_rule": (
            "flag when at least 3 of 5 nearest eligible training "
            "neighbours support a different label"
        ),
        "sensitivity_rule": (
            "flag when at least 4 of 5 nearest eligible training "
            "neighbours support a different label"
        ),
        "condition_E_reference": (
            "pre-existing legitimate training fold; injected "
            "candidates are not used as reference neighbours"
        ),
        "held_out_test_use_in_defence": "none",
    }
    (outdir / "defence_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    show = primary_summary[
        (primary_summary["condition"] == "E")
        & (primary_summary["source_rate_pct"].isin([30.0, 70.0, 100.0]))
    ][[
        "source_lineage",
        "destination_lineage",
        "source_rate_pct",
        "detection_sensitivity_mean_pct",
        "flag_precision_mean_pct",
        "false_positive_rate_mean_pct",
    ]]

    print("\nPrimary Condition E results:")
    print(show.to_string(index=False))
    print(f"\nOutputs: {outdir}")


if __name__ == "__main__":
    main()
