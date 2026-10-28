"""Analyses behind the numbers quoted in the README and the write-ups (no training here).

Each function prints one result; all of them only read the Kaggle data and the saved records in
experiments/. Usage:  python -m src.analysis [eda|errors|size|ensembles|meta|leaks|pseudo|figures|all]
"""
import hashlib
import itertools
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from src.classical import meta_features
from src.config import CLASSES, EXPERIMENTS_DIR, ROOT, SEED, SUBMISSIONS_DIR, TEST_DIR, TRAIN_DIR
from src.data import load_labels, preprocess, raw_metadata, split_indices, test_ids

PREDS = EXPERIMENTS_DIR / "preds"


def _val() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = load_labels()
    y = df["y"].to_numpy()
    _, va = split_indices(y)
    return df, y, va


def _p(exp: str, split: str = "val") -> np.ndarray:
    return np.load(PREDS / f"{exp}_{split}.npy")


def _avg(exps: list[str], split: str = "val") -> np.ndarray:
    return sum(_p(e, split) for e in exps) / len(exps)


def eda() -> None:
    """Source size, PNG mode and the colour hidden under transparent pixels, per vendor."""
    df = load_labels()
    meta = raw_metadata(TRAIN_DIR, df["Id"].tolist()).assign(label=df["Label"].values)
    meta["size"] = meta["w"].astype(str) + "x" + meta["h"].astype(str)
    meta["hidden"] = list(zip(meta.hid_r, meta.hid_g, meta.hid_b))
    print(df["Label"].value_counts().to_string(), "\n")
    print(pd.crosstab(meta.label, meta["size"]).loc[:, lambda d: d.sum() > 20], "\n")
    print(pd.crosstab(meta.label, meta.pil_mode), "\n")
    print(pd.crosstab(meta.label, meta.hidden.astype(str)).loc[:, lambda d: d.sum() > 50])


def errors(exps: tuple[str, ...] = ("E6b_convnext_tiny_finetune", "E8_convnext_tiny_224")) -> None:
    """Error analysis of the E9 ensemble (E6b + E8) on the validation split."""
    df, y, va = _val()
    p = _avg(list(exps))
    pred, yv = p.argmax(1), y[va]
    err = np.where(pred != yv)[0]
    conf = p.max(1)
    meta = raw_metadata(TRAIN_DIR, df["Id"].iloc[va].tolist())
    print(f"errors: {len(err)} of {len(va)}")
    print("top confusions (true -> pred):", Counter((CLASSES[yv[i]], CLASSES[pred[i]]) for i in err).most_common(8))
    print(f"mean confidence: correct {conf[pred == yv].mean():.3f}, wrong {conf[err].mean():.3f}; "
          f"wrong with conf > 0.9: {(conf[err] > 0.9).sum()}")
    print("error file sizes:", Counter(zip(meta.w.iloc[err], meta.h.iloc[err])).most_common(5))


def size() -> None:
    """Validation accuracy split by the original file size (72 px vs 160/240 px)."""
    df, y, va = _val()
    small = (raw_metadata(TRAIN_DIR, df["Id"].iloc[va].tolist()).w < 100).to_numpy()
    for e in ["E6b_convnext_tiny_finetune", "E12a_convnext_tiny_144_bicubic", "E12b_convnext_tiny_144_nearest",
              "E8_convnext_tiny_224", "E13_convnext_tiny_216_nearest", "E16_convnext_tiny_144_nearest_pseudo"]:
        c = _p(e).argmax(1) == y[va]
        print(f"{e:40s} all {c.mean():.4f}  72px (n={small.sum()}) {c[small].mean():.4f}  "
              f"large (n={(~small).sum()}) {c[~small].mean():.4f}")


def ensembles() -> None:
    """Every soft-vote combination of the validated ConvNeXt runs (and the weak-member check)."""
    _, y, va = _val()
    pool = ["E6b_convnext_tiny_finetune", "E8_convnext_tiny_224", "E12b_convnext_tiny_144_nearest",
            "E13_convnext_tiny_216_nearest"]
    for r in (2, 3, 4):
        for combo in itertools.combinations(pool, r):
            print(f"{(_avg(list(combo)).argmax(1) == y[va]).mean():.4f}", " + ".join(c.split("_")[0] for c in combo))
    for combo in (["E5_resnet18_finetune", "E6b_convnext_tiny_finetune", "E8_convnext_tiny_224"],
                  ["E3_simple_cnn_white", "E5_resnet18_finetune", "E6b_convnext_tiny_finetune", "E8_convnext_tiny_224"]):
        print(f"{(_avg(combo).argmax(1) == y[va]).mean():.4f}", " + ".join(c.split("_")[0] for c in combo), "(weak members)")


def meta() -> None:
    """Does stacking the metadata-only random forest onto the E9 ensemble help?"""
    df, y, va = _val()
    tr = np.setdiff1d(np.arange(len(y)), va)
    X = meta_features(raw_metadata(TRAIN_DIR, df["Id"].tolist()))
    pm = RandomForestClassifier(300, random_state=SEED, n_jobs=-1).fit(X[tr], y[tr]).predict_proba(X[va])
    p = _avg(["E6b_convnext_tiny_finetune", "E8_convnext_tiny_224"])
    for w in (0.0, 0.1, 0.2, 0.3, 0.5):
        print(f"metadata weight {w}: val acc {(((1 - w) * p + w * pm).argmax(1) == y[va]).mean():.4f}")


def leaks() -> None:
    """Exact duplicates (train/train and test/train) and whether file order carries the label."""
    df = load_labels()
    h = lambda f: hashlib.md5(np.asarray(Image.open(f).convert("RGBA")).tobytes()).hexdigest()  # noqa: E731
    df["h"] = [h(TRAIN_DIR / f"{i}.png") for i in df["Id"]]
    d = df[df.duplicated("h", keep=False)]
    print(f"train images with a duplicate in train: {len(d)}; groups with conflicting labels: "
          f"{int((d.groupby('h').Label.nunique() > 1).sum())}")
    t = pd.DataFrame({"Id": test_ids()})
    t["h"] = [h(TEST_DIR / f"{i}.png") for i in t["Id"]]
    lab = df.groupby("h").Label.first()
    m = t[t.h.isin(lab.index)]
    sub = pd.read_csv(SUBMISSIONS_DIR / "E15_ens_full_3models.csv", dtype={"Id": str}).set_index("Id")
    print(f"test images that are exact copies of a train image: {len(m)}; "
          f"E15 agrees with the train label on {(sub.loc[m.Id, 'Label'].values == m.h.map(lab).values).mean():.4f}")
    y = df["y"].to_numpy()
    prior = np.bincount(y) / len(y)
    print(f"consecutive ids with the same label: {(y[1:] == y[:-1]).mean():.3f} (chance {(prior ** 2).sum():.3f})")


def pseudo(source: str = "E13v_ens_val_3models", thr: float = 0.99) -> None:
    """Coverage and accuracy of the pseudo-label threshold on validation, and test images selected."""
    _, y, va = _val()
    pv, pt = _p(source), _p(source, "test")
    k = pv.max(1) >= thr
    print(f"val: conf >= {thr} covers {k.mean():.3f} of images at accuracy {(pv[k].argmax(1) == y[va][k]).mean():.4f}")
    print(f"test images selected: {(pt.max(1) >= thr).sum()}")


def figures() -> None:
    """Figures of report/ (grayscale): accuracy progression, training curves, error grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    rec = lambda e: json.loads((EXPERIMENTS_DIR / f"{e}.json").read_text())  # noqa: E731

    rows = [("E1a metadata only (RF)", "E1a_metadata_only"), ("E1b colour hist + HOG, SVM", "E1b_hog_color_svm"),
            ("E4 ResNet18 frozen", "E4_resnet18_frozen"), ("E2 small CNN, naive RGB", "E2_simple_cnn_naive"),
            ("E3 small CNN, white bg", "E3_simple_cnn_white"), ("E6 EfficientNet-B0", "E6_effnetb0_finetune"),
            ("E5 ResNet18 fine-tune", "E5_resnet18_finetune"), ("E7 ConvNeXt-T 128 + aug", "E7_convnext_tiny_aug"),
            ("E6b ConvNeXt-T 128", "E6b_convnext_tiny_finetune"), ("E12b ConvNeXt-T 144 nearest", "E12b_convnext_tiny_144_nearest"),
            ("E13 ConvNeXt-T 216 nearest", "E13_convnext_tiny_216_nearest"), ("E8 ConvNeXt-T 224", "E8_convnext_tiny_224"),
            ("E16 E12b + pseudo-labels", "E16_convnext_tiny_144_nearest_pseudo"), ("E9 ensemble E6b+E8", "E9_ens_convnext128_224"),
            ("E13v ensemble E8+E12b+E13", "E13v_ens_val_3models")]
    val = [round(rec(e)["val_acc"], 3) for _, e in rows]
    kag = [rec(e)["kaggle_public"] if e in ("E3_simple_cnn_white", "E8_convnext_tiny_224", "E9_ens_convnext128_224") else None
           for _, e in rows]
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    ax.barh(range(len(rows)), val, color="0.78", edgecolor="black", linewidth=0.6, label="validation (20% hold-out)")
    ax.scatter([k for k in kag if k], [i for i, k in enumerate(kag) if k], marker="D", color="black", s=16, zorder=3,
               label="Kaggle public")
    for i, v in enumerate(val):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=7.5, bbox=dict(fc="white", ec="none", pad=0.3))
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.6, 1.0)
    ax.set_xlabel("accuracy")
    ax.axvline(0.788, color="black", ls="--", lw=0.7, label="team's earlier best (0.788)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.13), ncol=3, frameon=False, fontsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "accuracy_progression.pdf")

    runs = [("E3_simple_cnn_white", "E3 small CNN (scratch)", "-", "o"), ("E4_resnet18_frozen", "E4 ResNet18 frozen", ":", "s"),
            ("E6b_convnext_tiny_finetune", "E6b ConvNeXt-T 128", "--", "^"), ("E7_convnext_tiny_aug", "E7 ConvNeXt-T 128 + aug", "-.", "v"),
            ("E8_convnext_tiny_224", "E8 ConvNeXt-T 224", "-", "D")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for e, lab, ls, mk in runs:
        h = rec(e)["extra"]["history"]
        ep = [x["epoch"] for x in h]
        a.plot(ep, [x["loss"] for x in h], ls=ls, marker=mk, ms=2.5, color="black", lw=0.9, label=lab)
        b.plot(ep, [x["val_acc"] for x in h], ls=ls, marker=mk, ms=2.5, color="black", lw=0.9, label=lab)
    a.set_yscale("log")
    a.set(xlabel="epoch", ylabel="training loss (log)")
    b.set(xlabel="epoch", ylabel="validation accuracy", ylim=(0.3, 1.0))
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    hd, lb = b.get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    plt.tight_layout(rect=(0, 0.13, 1, 1))
    plt.savefig(out / "training_curves.pdf")

    df, y, va = _val()
    p = _avg(["E6b_convnext_tiny_finetune", "E8_convnext_tiny_224"])
    wrong = np.where(p.argmax(1) != y[va])[0]
    wrong = wrong[np.argsort(-p[wrong].max(1))][:24]
    fig, axs = plt.subplots(4, 6, figsize=(13, 9.5))
    for ax, j in zip(axs.flat, wrong):
        g = va[j]
        im = Image.open(TRAIN_DIR / f"{df['Id'].iloc[g]}.png")
        ax.imshow(preprocess(im, "white", 224))
        ax.axis("off")
        ax.set_title(f"true {CLASSES[y[g]]}\npred {CLASSES[p[j].argmax()]} ({p[j].max():.2f}) {im.size[0]}px", fontsize=9)
    plt.suptitle("E9 validation errors, most confident first", fontsize=12)
    plt.tight_layout()
    plt.savefig(out / "e9_val_errors.png", dpi=90)
    print(f"figures written to {out}")


TASKS = {"eda": eda, "errors": errors, "size": size, "ensembles": ensembles, "meta": meta, "leaks": leaks,
         "pseudo": pseudo, "figures": figures}

if __name__ == "__main__":
    names = sys.argv[1:] or ["all"]
    for n in (TASKS if names == ["all"] else names):
        print(f"\n## {n}")
        TASKS[n]()
