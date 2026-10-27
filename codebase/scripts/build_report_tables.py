"""Collect every number the report cites into report/numbers.json and draw the experiment-ladder chart.

Reads outputs/eda, outputs/cpu, outputs/convnext224 (if present) and report/reported_experiments.json.
Nothing is typed by hand here: if a file is missing, the corresponding entry is null and the report shows "pending".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import os
OUT = Path(os.environ.get("EMOJI_OUTPUTS", ROOT / "outputs"))
REP = ROOT / "report"
REP.mkdir(exist_ok=True)
(REP / "figures").mkdir(exist_ok=True)
GPU_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "convnext224"


def load_json(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


N = {"generated_from": {}, "eda": None, "cpu": None, "gpu": None, "reported": load_json(REP / "reported_experiments.json"),
     "public_lb": load_json(REP / "public_leaderboard.json")}

# ------------------------------------------------------------------ EDA
eda = load_json(OUT / "eda" / "summary.json")
if eda:
    size_tab = pd.read_csv(OUT / "eda" / "size_by_provider.csv", index_col=0)
    mode_tab = pd.read_csv(OUT / "eda" / "mode_by_provider.csv", index_col=0)
    under = pd.read_csv(OUT / "eda" / "colour_under_transparency_by_provider.csv", index_col=0)
    shift = pd.read_csv(OUT / "eda" / "size_mode_train_vs_test.csv", index_col=0)
    # modal native size per provider and its share
    modal = {}
    for prov, row in size_tab.iterrows():
        s = row.idxmax()
        modal[prov] = {"size": s, "count": int(row.max()), "share": float(row.max() / row.sum()),
                       "mode": mode_tab.loc[prov].idxmax(), "mode_share": float(mode_tab.loc[prov].max() / mode_tab.loc[prov].sum())}
    N["eda"] = {**eda, "modal": modal, "size_table": size_tab.to_dict(), "mode_table": mode_tab.to_dict(),
                "under_colour": under.round({"under_r": 1, "under_g": 1, "under_b": 1, "under_std": 1, "frac_transparent": 3}).to_dict(orient="index"),
                "train_test_shift_rows": shift.round(4).reset_index().rename(columns={"index": "size_mode"}).to_dict(orient="records")}
    N["generated_from"]["eda"] = str(OUT / "eda" / "summary.json")

# ------------------------------------------------------------------ CPU ladder
cpu = load_json(OUT / "cpu" / "results.json")
if cpu:
    N["cpu"] = cpu
    N["cpu"]["duplicates"] = load_json(OUT / "cpu" / "duplicates.json")
    for tag in ("emojicnnv2", "emojicnnv2_fusion"):
        h = OUT / "cpu" / f"{tag}_history.csv"
        if h.is_file():
            df = pd.read_csv(h)
            N["cpu"][f"{tag}_history"] = {"epochs": int(len(df)), "final_train_accuracy": float(df["train_accuracy"].iloc[-1]),
                                          "best_val_accuracy": float(df["val_accuracy"].max()),
                                          "seconds_per_epoch": float(df["seconds"].mean())}
    f = OUT / "cpu" / "failures_emojicnnv2.csv"
    if f.is_file():
        N["cpu"]["top_failures"] = pd.read_csv(f, dtype={"Id": str}).head(8).to_dict(orient="records")
    N["generated_from"]["cpu"] = str(OUT / "cpu" / "results.json")

# ------------------------------------------------------------------ error structure by native-size family
def family_of(w, h):
    m = max(int(w), int(h))
    return "72 px family" if m <= 108 else "160 px family" if m <= 160 else "240 px family"


def family_breakdown(y_true, proba, meta_rows, class_names):
    """Accuracy inside each native-resolution family and how many errors cross families."""
    import collections
    fam = np.array([family_of(w, h) for w, h in zip(meta_rows["width"], meta_rows["height"])])
    pred = proba.argmax(1)
    # family of the *predicted* provider = that provider's modal family
    modal_fam = {}
    for c in class_names:
        sub = meta_rows[meta_rows["Label"] == c]
        modal_fam[c] = collections.Counter(family_of(w, h) for w, h in zip(sub["width"], sub["height"])).most_common(1)[0][0]
    out = {}
    for f in ("72 px family", "160 px family", "240 px family"):
        m = fam == f
        if m.any():
            out[f] = {"n": int(m.sum()), "accuracy": float((pred[m] == y_true[m]).mean())}
    wrong = pred != y_true
    cross = sum(modal_fam[class_names[p_]] != fam[i] for i, p_ in zip(np.flatnonzero(wrong), pred[wrong]))
    out["errors_total"] = int(wrong.sum()); out["errors_crossing_family"] = int(cross)
    return out


if cpu:
    split = pd.read_csv(ROOT / "data" / "split_seed42.csv", dtype=str)
    meta_tr = pd.read_csv(OUT / "cpu" / "metadata_train.csv", dtype={"Id": str}).merge(split, on="Id")
    labels_sorted = split.sort_values("Id").reset_index(drop=True)
    C = sorted(labels_sorted["Label"].unique())
    val_mask = labels_sorted["split"].to_numpy() == "validation"
    yv = labels_sorted.loc[val_mask, "Label"].map({c: i for i, c in enumerate(C)}).to_numpy()
    meta_val = meta_tr.set_index("Id").loc[labels_sorted.loc[val_mask, "Id"]].reset_index()
    fam_tab = {}
    for key, fname in (("Baseline CNN (30 Sept)", "val_proba_baseline.npy"), ("EmojiCNNv2 (pixels)", "val_proba_emojicnnv2.npy"),
                       ("EmojiCNNv2 + metadata", "val_proba_emojicnnv2_fusion.npy")):
        f = OUT / "cpu" / fname
        if f.is_file():
            pv = np.load(f)
            if len(pv) == len(yv):
                fam_tab[key] = family_breakdown(yv, pv, meta_val, C)
    N["cpu"]["family_breakdown"] = fam_tab

# ------------------------------------------------------------------ GPU final model
gpu = load_json(GPU_DIR / "oof_summary.json")
if gpu:
    N["gpu"] = gpu
    N["gpu"]["config"] = load_json(GPU_DIR / "config.json")
    fs = GPU_DIR / "fold_summary.csv"
    if fs.is_file():
        N["gpu"]["fold_table"] = pd.read_csv(fs).round(4).to_dict(orient="records")
    f = GPU_DIR / "failures_oof.csv"
    if f.is_file():
        N["gpu"]["top_failures"] = pd.read_csv(f, dtype={"Id": str}).head(8).to_dict(orient="records")
    sub = GPU_DIR / "submission.csv"
    if sub.is_file():
        N["gpu"]["submission_label_counts"] = pd.read_csv(sub, dtype=str)["Label"].value_counts().to_dict()
    # confusion pairs from oof
    oof = np.load(GPU_DIR / "oof_proba.npy") if (GPU_DIR / "oof_proba.npy").is_file() else None
    if oof is not None:
        from sklearn.metrics import confusion_matrix
        labels = pd.read_csv(ROOT / "data" / "extracted" / "train_labels.csv", dtype=str).sort_values("Id")
        C = sorted(labels["Label"].unique())
        y = labels["Label"].map({c: i for i, c in enumerate(C)}).to_numpy()
        ok = ~np.isnan(oof[:, 0])
        cm = confusion_matrix(y[ok], oof[ok].argmax(1), labels=range(len(C)))
        pairs = [(C[i], C[j], int(cm[i, j])) for i in range(len(C)) for j in range(len(C)) if i != j and cm[i, j] > 0]
        pairs.sort(key=lambda t: -t[2])
        N["gpu"]["top_confusions"] = [{"true": a, "pred": b, "count": c} for a, b, c in pairs[:8]]
        N["gpu"]["confusion_matrix"] = {"classes": C, "counts": cm.tolist()}
        meta_all = pd.read_csv(OUT / "cpu" / "metadata_train.csv", dtype={"Id": str}) if (OUT / "cpu" / "metadata_train.csv").is_file() else None
        if meta_all is not None:
            meta_all = meta_all.merge(labels, on="Id").set_index("Id").loc[labels["Id"]].reset_index()
            N["gpu"]["family_breakdown"] = family_breakdown(y[ok], oof[ok], meta_all[ok].reset_index(drop=True), C)
    N["generated_from"]["gpu"] = str(GPU_DIR / "oof_summary.json")

# ------------------------------------------------------------------ second GPU run and the cross-run ensemble
alt_dir = OUT / "convnext224_last"
if gpu and (alt_dir / "oof_summary.json").is_file():
    alt = load_json(alt_dir / "oof_summary.json")
    alt["fold_table"] = pd.read_csv(alt_dir / "fold_summary.csv").round(4).to_dict(orient="records") if (alt_dir / "fold_summary.csv").is_file() else None
    N["gpu_alt"] = alt
    oof_a = np.load(GPU_DIR / "oof_proba.npy"); oof_b = np.load(alt_dir / "oof_proba.npy")
    labels = pd.read_csv(ROOT / "data" / "extracted" / "train_labels.csv", dtype=str).sort_values("Id")
    C = sorted(labels["Label"].unique()); y = labels["Label"].map({c: i for i, c in enumerate(C)}).to_numpy()
    ok = ~np.isnan(oof_a[:, 0]) & ~np.isnan(oof_b[:, 0])
    from sklearn.metrics import f1_score
    mix = 0.5 * (oof_a + oof_b)
    pa, pb, pm = oof_a[ok].argmax(1), oof_b[ok].argmax(1), mix[ok].argmax(1)
    N["gpu_ensemble"] = {"n": int(ok.sum()),
                         "accuracy_a": float((pa == y[ok]).mean()), "accuracy_b": float((pb == y[ok]).mean()),
                         "accuracy_mean": float((pm == y[ok]).mean()), "macro_f1_mean": float(f1_score(y[ok], pm, average="macro")),
                         "both_right": int(((pa == y[ok]) & (pb == y[ok])).sum()), "only_a_right": int(((pa == y[ok]) & (pb != y[ok])).sum()),
                         "only_b_right": int(((pa != y[ok]) & (pb == y[ok])).sum()), "both_wrong": int(((pa != y[ok]) & (pb != y[ok])).sum()),
                         "test_agreement_a_b": None}
    ta, tb = GPU_DIR / "test_proba_mean.npy", alt_dir / "test_proba_mean.npy"
    if ta.is_file() and tb.is_file():
        A, B = np.load(ta), np.load(tb)
        N["gpu_ensemble"]["test_agreement_a_b"] = float((A.argmax(1) == B.argmax(1)).mean())
        N["gpu_ensemble"]["test_changed_by_mean_vs_a"] = int(((0.5 * (A + B)).argmax(1) != A.argmax(1)).sum())
        N["gpu_ensemble"]["test_changed_by_mean_vs_b"] = int(((0.5 * (A + B)).argmax(1) != B.argmax(1)).sum())
    N["generated_from"]["gpu_alt"] = str(alt_dir / "oof_summary.json")
    # ---- the final model = run B (last-epoch checkpoints, 5 folds): metrics, confusions, family breakdown and figures
    # built locally from its out-of-fold probabilities. The A+B average stays in N["gpu_ensemble"] as an ablation.
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from emoji_provider import evaluate as E
    from emoji_provider import data as D
    mix = oof_b
    final = E.summarize(y[ok], mix[ok], C)
    final["source"] = "run B: convnext224_last (5 folds, last-epoch checkpoints, flip TTA)"
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y[ok], mix[ok].argmax(1), labels=range(len(C)))
    pairs = sorted([(C[i], C[j], int(cm[i, j])) for i in range(len(C)) for j in range(len(C)) if i != j and cm[i, j] > 0], key=lambda t: -t[2])
    final["top_confusions"] = [{"true": a, "pred": b, "count": c} for a, b, c in pairs[:8]]
    final["confusion_matrix"] = {"classes": C, "counts": cm.tolist()}
    meta_all = pd.read_csv(OUT / "cpu" / "metadata_train.csv", dtype={"Id": str}).merge(labels, on="Id").set_index("Id").loc[labels["Id"]].reset_index()
    final["family_breakdown"] = family_breakdown(y[ok], mix[ok], meta_all[ok].reset_index(drop=True), C)
    if (GPU_DIR / "../../data/split_seed42.csv").resolve().is_file() or (ROOT / "data" / "split_seed42.csv").is_file():
        split = pd.read_csv(ROOT / "data" / "split_seed42.csv", dtype=str).set_index("Id").loc[labels["Id"]]
        vm = (split["split"].to_numpy() == "validation") & ok
        fs = E.summarize(y[vm], mix[vm], C)
        final["stored_split_validation_oof"] = {"accuracy": fs["accuracy"], "macro_f1": fs["macro_f1"], "n": fs["n"], "correct": fs["accuracy_correct"]}
    E.plot_confusion(y[ok], mix[ok].argmax(1), C, REP / "figures" / "cm_final_oof.png", f"Final model (5 folds, last-epoch checkpoints), out-of-fold: {final['accuracy']:.2%}")
    E.plot_confusion(y[ok], mix[ok].argmax(1), C, REP / "figures" / "cm_final_oof_norm.png", "Final model, out-of-fold (row-normalised)", normalize=True)
    cache96 = ROOT / "data" / "cache_train_96.npy"
    if cache96.is_file():
        X96 = np.load(cache96, mmap_mode="r")
        ids = labels["Id"].to_numpy()
        fails = E.plot_failures(np.asarray(X96[ok]), y[ok], mix[ok], C, ids[ok], REP / "figures" / "failures_final.png",
                                "Final model: most confident out-of-fold mistakes (true→predicted, confidence)")
        fails.to_csv(REP / "failures_final.csv", index=False)
        final["top_failures"] = fails.head(8).to_dict(orient="records")
    N["final"] = final

# ------------------------------------------------------------------ ladder chart (verified + reported)
rows = []
if cpu:
    rows.append(("Majority class", cpu["majority_class"]["accuracy"], "verified"))
    fmt = max((k for k in cpu if k.startswith("metadata-only") and k.split("/")[1].strip() == "size+mode"),
              key=lambda k: cpu[k]["accuracy"], default=None)
    if fmt:
        rows.append((f"File format only (dimensions + PNG mode), no pixels\n({fmt.split('/')[-1].strip()})", cpu[fmt]["accuracy"], "verified"))
    best_meta = max((k for k in cpu if k.startswith("metadata-only")), key=lambda k: cpu[k]["accuracy"], default=None)
    if best_meta and best_meta != fmt:
        rows.append((f"File metadata + colour under transparency, no drawing\n({best_meta.split('/')[-1].strip()})", cpu[best_meta]["accuracy"], "verified"))
    if "baseline_sept30 / EmojiCNN 96px" in cpu:
        rows.append(("Baseline CNN, 30 Sept (96 px)", cpu["baseline_sept30 / EmojiCNN 96px"]["accuracy"], "verified"))
    if "EmojiCNNv2 96px / pixels only" in cpu:
        rows.append(("EmojiCNNv2 from scratch (96 px)", cpu["EmojiCNNv2 96px / pixels only"]["accuracy"], "verified"))
    if "EmojiCNNv2 96px / pixels + file metadata" in cpu:
        rows.append(("EmojiCNNv2 + file metadata (ablation)", cpu["EmojiCNNv2 96px / pixels + file metadata"]["accuracy"], "verified"))
if N["reported"]:
    for e in N["reported"]["experiments"]:
        if e["id"] in ("E1b_hog_color_svm", "E4_resnet18_frozen", "E5_resnet18_finetune", "E6b_convnext_tiny_finetune",
                       "E7_convnext_tiny_aug", "E8_convnext_tiny_224", "E9_ens_convnext128_224"):
            label = {"E1b_hog_color_svm": "HOG + colour histogram, SVM", "E4_resnet18_frozen": "ResNet18 frozen (128 px)",
                     "E5_resnet18_finetune": "ResNet18 fine-tuned (128 px)", "E6b_convnext_tiny_finetune": "ConvNeXt-Tiny (128 px)",
                     "E7_convnext_tiny_aug": "ConvNeXt-Tiny (128 px) + rotation/scale aug", "E8_convnext_tiny_224": "ConvNeXt-Tiny (224 px)",
                     "E9_ens_convnext128_224": "Ensemble ConvNeXt 128 + 224"}[e["id"]]
            rows.append((label, e["val_accuracy"], "reported"))
if gpu:
    rows.append((f"ConvNeXt-Tiny 224 px, 5-fold + TTA, best-val-loss epoch\n(out-of-fold, {gpu['covered_images']:,} images)", gpu["accuracy"], "verified"))
if N.get("gpu_alt"):
    rows.append((f"ConvNeXt-Tiny 224 px, 5-fold + TTA, last epoch\n(out-of-fold, {N['gpu_alt']['covered_images']:,} images)", N["gpu_alt"]["accuracy"], "verified"))
if N.get("gpu_ensemble"):
    rows.append((f"Both checkpoint sets averaged (10 checkpoints)\n(out-of-fold, {N['gpu_ensemble']['n']:,} images)", N["gpu_ensemble"]["accuracy_mean"], "verified"))
if rows:
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.2))
    colors = {"verified": "#1f5fbf", "reported": "#b0b7c3"}
    ax.barh(range(len(rows)), [r[1] for r in rows], color=[colors[r[2]] for r in rows])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.004, i, f"{r[1]:.1%}", va="center", fontsize=8)
    ax.set_xlim(0, 1.08); ax.set_xlabel("validation accuracy")
    ax.axvline(1 / 7, color="grey", ls=":", lw=1); ax.text(1 / 7 + 0.004, len(rows) - 0.4, "chance", fontsize=7, color="grey")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors["verified"], label="verified in this codebase (stored split or out-of-fold)"),
                       Patch(color=colors["reported"], label="reported by teammate (13 Oct log, 80/20 split)")],
              frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Experiment ladder: what each change bought")
    fig.tight_layout(); fig.savefig(REP / "figures" / "ladder.png", dpi=200); plt.close(fig)
    N["ladder_rows"] = [{"label": r[0].replace("\n", " "), "accuracy": r[1], "status": r[2]} for r in rows]

# ---- learning curves of the final model: every fold on one figure (from the fold*_history.csv files).
# Runs A and B share seeds; if their histories coincide the figure shows one trajectory per fold with both kept epochs.
run_dirs = [d for d in (GPU_DIR, alt_dir) if list(d.glob("fold*_history.csv"))]
if run_dirs:
    hist_a = {k: pd.read_csv(GPU_DIR / f"fold{k}_history.csv") for k in range(10) if (GPU_DIR / f"fold{k}_history.csv").is_file()}
    hist_b = {k: pd.read_csv(alt_dir / f"fold{k}_history.csv") for k in range(10) if (alt_dir / f"fold{k}_history.csv").is_file()}
    metric_cols = ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]
    identical = bool(hist_b) and all(k in hist_b and np.array_equal(hist_a[k][metric_cols].to_numpy(), hist_b[k][metric_cols].to_numpy())
                                     for k in hist_a)
    kept_a = {k: (load_json(GPU_DIR / f"fold{k}_run.json") or {}).get("best_epoch") for k in hist_a}
    kept_b = {k: (load_json(alt_dir / f"fold{k}_run.json") or {}).get("best_epoch") for k in hist_b}
    panels = [(GPU_DIR, hist_a, kept_a, "run A" if not identical else "runs A and B (identical trajectories)")]
    if hist_b and not identical:
        panels.append((alt_dir, hist_b, kept_b, "run B"))
    fig, axes = plt.subplots(len(panels), 3, figsize=(12, 3.4 * len(panels)), squeeze=False)
    curve_stats = {"identical_trajectories": identical}
    for r, (d, hists, kept, lbl) in enumerate(panels):
        for k, hist in hists.items():
            c = plt.cm.tab10(k)
            axes[r, 0].plot(hist["epoch"], hist["train_loss"], color=c, ls="--", lw=1)
            axes[r, 0].plot(hist["epoch"], hist["val_loss"], color=c, lw=1.4, label=f"fold {k}")
            axes[r, 1].plot(hist["epoch"], hist["val_accuracy"], color=c, lw=1.4, label=f"fold {k}")
            axes[r, 2].plot(hist["epoch"], hist["train_accuracy"], color=c, lw=1.4)
            if kept.get(k):
                e = kept[k]
                axes[r, 1].plot([e], [float(hist.loc[hist["epoch"] == e, "val_accuracy"].iloc[0])], "o", color=c, ms=5)
                axes[r, 0].plot([e], [float(hist.loc[hist["epoch"] == e, "val_loss"].iloc[0])], "o", color=c, ms=5)
        H = pd.concat(hists.values())
        by_epoch = H.groupby("epoch").mean(numeric_only=True)
        curve_stats[d.name] = {"mean_val_loss_by_epoch": by_epoch["val_loss"].round(4).to_dict(),
                               "mean_val_accuracy_by_epoch": by_epoch["val_accuracy"].round(4).to_dict(),
                               "mean_train_accuracy_by_epoch": by_epoch["train_accuracy"].round(4).to_dict(),
                               "epoch_of_min_mean_val_loss": int(by_epoch["val_loss"].idxmin()),
                               "epoch_of_max_mean_val_accuracy": int(by_epoch["val_accuracy"].idxmax()),
                               "kept_epochs": kept}
        last = int(H["epoch"].max())
        if identical or d == alt_dir:
            for ax in axes[r, :2]:
                ax.axvline(last, color="grey", ls=":", lw=1)
        axes[r, 0].set_title("cross-entropy (solid: validation, dashed: train)", fontsize=9)
        axes[r, 1].set_title("validation accuracy (dot: epoch kept by run A, dotted line: by run B)", fontsize=9)
        axes[r, 2].set_title("training accuracy", fontsize=9)
        axes[r, 0].set_yscale("log"); axes[r, 1].set_ylim(0.94, 0.985); axes[r, 2].set_ylim(0.9, 1.001)
        axes[r, 1].text(1.1, 0.9415, "epoch 1: 0.88 to 0.93, off scale", fontsize=7, color="grey")
        for ax in axes[r]:
            ax.set_xlabel("epoch"); ax.set_xticks(range(1, last + 1)); ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        axes[r, 1].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("ConvNeXt-Tiny 224 px, 5 folds (OneCycle, 10 epochs, batch 32, AdamW 2e-4)"
                 + (": runs A and B followed identical trajectories" if identical else ""), fontsize=10)
    fig.tight_layout(); fig.savefig(REP / "figures" / "curves_convnext_runs.png", dpi=200); plt.close(fig)
    N["gpu_curves"] = curve_stats

(REP / "numbers.json").write_text(json.dumps(N, indent=2, default=float))
print("wrote", REP / "numbers.json")
print("sections present:", {k: (v is not None) for k, v in N.items() if k in ("eda", "cpu", "gpu", "reported")})
