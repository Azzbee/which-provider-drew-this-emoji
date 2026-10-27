"""Exploratory data analysis on the raw PNG files (sizes, modes, transparency, duplicates, figures)."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def run_eda(data_dir: Path, out_dir: Path) -> dict:
    """Scan every train/test PNG once; write metadata CSVs, summary tables, summary.json and figures.

    Returns the summary dict. Takes about a minute for the 19,758 files.
    """
    DATA, OUT = Path(data_dir), Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG = OUT / "figures"
    FIG.mkdir(exist_ok=True)
    labels = pd.read_csv(DATA / "train_labels.csv", dtype={"Id": str, "Label": str})

    class_names = sorted(labels["Label"].unique())
    print("classes:", class_names)


    def scan(path: Path) -> dict:
        raw = path.read_bytes()
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode
            info_transparency = "transparency" in im.info
            rgba = im.convert("RGBA")
            arr = np.asarray(rgba)
        alpha = arr[..., 3]
        transparent = alpha == 0
        frac_transparent = float(transparent.mean())
        if transparent.any():
            under = arr[transparent][:, :3].astype(np.float64)
            under_mean = under.mean(axis=0)
            under_std = under.std(axis=0).mean()
        else:
            under_mean = np.array([np.nan] * 3)
            under_std = np.nan
        # Preprocessed pixel hash (same recipe as the training pipeline: white, contain, pad 96)
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(white, rgba).convert("RGB")
        fitted = ImageOps.contain(rgb, (96, 96), Image.Resampling.LANCZOS)
        square = Image.new("RGB", (96, 96), (255, 255, 255))
        square.paste(fitted, ((96 - fitted.width) // 2, (96 - fitted.height) // 2))
        pix_hash = hashlib.sha256(np.asarray(square, dtype=np.uint8).tobytes()).hexdigest()
        return {
            "Id": path.stem,
            "width": w,
            "height": h,
            "mode": mode,
            "has_alpha": mode in ("RGBA", "LA") or info_transparency,
            "frac_transparent": frac_transparent,
            "under_r": under_mean[0],
            "under_g": under_mean[1],
            "under_b": under_mean[2],
            "under_std": under_std,
            "bytes": len(raw),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "pix96_sha256": pix_hash,
        }


    rows = {}
    for split in ("train", "test"):
        files = sorted((DATA / split).glob("*.png"))
        recs = []
        for i, f in enumerate(files):
            recs.append(scan(f))
            if (i + 1) % 2000 == 0:
                print(f"{split}: {i + 1}/{len(files)}", flush=True)
        df = pd.DataFrame(recs)
        if split == "train":
            df = df.merge(labels, on="Id", how="left")
            assert df["Label"].notna().all()
        df.to_csv(OUT / f"metadata_{split}.csv", index=False)
        rows[split] = df
        print(split, df.shape)

    tr, te = rows["train"], rows["test"]

    # ---------------------------------------------------------------- summary tables
    summary = {}
    summary["n_train"] = int(len(tr))
    summary["n_test"] = int(len(te))
    summary["class_counts"] = tr["Label"].value_counts().to_dict()
    summary["class_share"] = (tr["Label"].value_counts(normalize=True).round(4)).to_dict()
    summary["majority_share"] = float(tr["Label"].value_counts(normalize=True).max())
    summary["chance"] = 1 / len(class_names)

    tr["size"] = tr["width"].astype(str) + "x" + tr["height"].astype(str)
    te["size"] = te["width"].astype(str) + "x" + te["height"].astype(str)
    size_by_label = pd.crosstab(tr["Label"], tr["size"])
    mode_by_label = pd.crosstab(tr["Label"], tr["mode"])
    size_by_label.to_csv(OUT / "size_by_provider.csv")
    mode_by_label.to_csv(OUT / "mode_by_provider.csv")
    print("\nNative size by provider:\n", size_by_label)
    print("\nMode by provider:\n", mode_by_label)

    # Train vs test distribution of (size, mode): is the test set drawn from the same export pipeline?
    sm_tr = (tr["size"] + "/" + tr["mode"]).value_counts(normalize=True)
    sm_te = (te["size"] + "/" + te["mode"]).value_counts(normalize=True)
    shift = pd.concat([sm_tr.rename("train"), sm_te.rename("test")], axis=1).fillna(0)
    shift.to_csv(OUT / "size_mode_train_vs_test.csv")
    print("\n(size/mode) share train vs test:\n", shift.round(4))
    summary["size_mode_total_variation_train_test"] = float(0.5 * (shift["train"] - shift["test"]).abs().sum())

    # Junk colour under transparent pixels, by provider
    under = tr.dropna(subset=["under_r"]).groupby("Label")[["under_r", "under_g", "under_b", "under_std", "frac_transparent"]].mean().round(2)
    under.to_csv(OUT / "colour_under_transparency_by_provider.csv")
    print("\nMean colour stored under fully transparent pixels:\n", under)

    # Duplicates
    raw_groups = tr.groupby("raw_sha256")["Label"].agg(["count", "nunique"])
    pix_groups = tr.groupby("pix96_sha256")["Label"].agg(["count", "nunique"])
    summary["raw_duplicate_images_beyond_first"] = int((raw_groups["count"] - 1).sum())
    summary["raw_conflicting_groups"] = int((raw_groups["nunique"] > 1).sum())
    summary["pix96_duplicate_images_beyond_first"] = int((pix_groups["count"] - 1).sum())
    summary["pix96_conflicting_groups"] = int((pix_groups["nunique"] > 1).sum())
    # Train/test leakage by exact bytes or by preprocessed pixels
    summary["test_images_identical_to_a_train_image_raw"] = int(te["raw_sha256"].isin(set(tr["raw_sha256"])).sum())
    summary["test_images_identical_to_a_train_image_pix96"] = int(te["pix96_sha256"].isin(set(tr["pix96_sha256"])).sum())
    # Within-test duplicates
    summary["test_duplicate_images_beyond_first_pix96"] = int((te.groupby("pix96_sha256").size() - 1).sum())

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    PALETTE = {"apple": "#1f77b4", "facebook": "#2ca02c", "google": "#d62728", "messenger": "#9467bd",
               "mozilla": "#ff7f0e", "samsung": "#17becf", "whatsapp": "#8c564b"}

    # 1. class distribution
    fig, ax = plt.subplots(figsize=(7, 3.4))
    counts = tr["Label"].value_counts().reindex(class_names)
    ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, f"{v}\n({v / len(tr):.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("training images")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("Class distribution of the 9,879 training images")
    fig.tight_layout()
    fig.savefig(FIG / "class_distribution.png", dpi=200)
    plt.close(fig)

    # 2. native size x provider heatmap (row-normalised)
    order_sizes = sorted(size_by_label.columns, key=lambda s: (int(s.split("x")[0]), int(s.split("x")[1])))
    norm = size_by_label[order_sizes].div(size_by_label.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order_sizes)))
    ax.set_xticklabels(order_sizes, rotation=45, ha="right")
    ax.set_yticks(range(len(norm.index)))
    ax.set_yticklabels(norm.index)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = size_by_label[order_sizes].values[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="white" if norm.values[i, j] > 0.5 else "black")
    ax.set_title("Native PNG dimensions per provider (counts; shading = share of the provider)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="share of provider")
    fig.tight_layout()
    fig.savefig(FIG / "native_size_by_provider.png", dpi=200)
    plt.close(fig)

    # 3. mode x provider heatmap
    norm_m = mode_by_label.div(mode_by_label.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    im = ax.imshow(norm_m.values, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(mode_by_label.shape[1]))
    ax.set_xticklabels(mode_by_label.columns)
    ax.set_yticks(range(mode_by_label.shape[0]))
    ax.set_yticklabels(mode_by_label.index)
    for i in range(norm_m.shape[0]):
        for j in range(norm_m.shape[1]):
            v = mode_by_label.values[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="white" if norm_m.values[i, j] > 0.5 else "black")
    ax.set_title("PNG colour mode per provider")
    fig.tight_layout()
    fig.savefig(FIG / "native_mode_by_provider.png", dpi=200)
    plt.close(fig)

    # 4. sample grid: 6 examples per provider, composited on white, native pixels (nearest upscaling to show resolution)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(class_names), 6, figsize=(9, 10.5))
    for r, c in enumerate(class_names):
        ids = tr.loc[tr["Label"] == c, "Id"].sample(6, random_state=r).tolist()
        for k, img_id in enumerate(ids):
            with Image.open(DATA / "train" / f"{img_id}.png") as im:
                rgba = im.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                rgb = Image.alpha_composite(white, rgba).convert("RGB")
            ax = axes[r, k]
            ax.imshow(rgb)
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(c, fontsize=11, rotation=0, ha="right", va="center", labelpad=8)
            ax.set_title(f"{rgb.width}x{rgb.height}", fontsize=7, pad=2)
    fig.suptitle("Random training examples per provider (alpha-composited on white; title = native size)", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "samples_per_provider.png", dpi=200)
    plt.close(fig)

    # 5. same emoji across providers? find pixel-identical duplicates across labels is 0; instead show the
    #    raw vs white-composited view to illustrate the transparency artefact
    # pick one image with high transparency per provider
    fig, axes = plt.subplots(2, len(class_names), figsize=(12, 4))
    for j, c in enumerate(class_names):
        sub = tr[(tr["Label"] == c) & (tr["frac_transparent"] > 0.3)]
        img_id = sub.sort_values("frac_transparent", ascending=False)["Id"].iloc[len(sub) // 2]
        with Image.open(DATA / "train" / f"{img_id}.png") as im:
            rgba = im.convert("RGBA")
            naive = im.convert("RGB")            # what PIL/skimage default loading gives: alpha dropped
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            comp = Image.alpha_composite(white, rgba).convert("RGB")
        axes[0, j].imshow(naive); axes[0, j].set_title(c, fontsize=9)
        axes[1, j].imshow(comp)
        for a in axes[:, j]:
            a.set_xticks([]); a.set_yticks([])
    axes[0, 0].set_ylabel("alpha dropped\n(naive RGB)", fontsize=9)
    axes[1, 0].set_ylabel("alpha-composited\non white", fontsize=9)
    fig.suptitle("Why alpha handling matters: the colour hidden under transparent pixels differs by provider", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "alpha_naive_vs_white.png", dpi=200)
    plt.close(fig)

    print("figures written to", FIG)
    return summary

