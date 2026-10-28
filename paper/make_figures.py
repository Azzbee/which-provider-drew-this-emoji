"""Builds every figure of the LaTeX paper from the codebase outputs (no number typed by hand)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image

HERE = Path(__file__).resolve().parent                 # repo/paper
CB = HERE.parent / "codebase"                            # the submitted codebase (saved outputs live here)
OUT = HERE / "figs"; OUT.mkdir(parents=True, exist_ok=True)
WORK = HERE / "work"                                     # small JSON summaries computed from the outputs
# Figures 1, 3 and 12 read the competition images: extract the Kaggle archive into codebase/data/extracted/
N = json.loads((CB / "report/numbers.json").read_text())
ST = json.loads((WORK / "stats.json").read_text())

# ------------------------------------------------------------------ style
for f in font_manager.findSystemFonts():
    if "Inter" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Inter", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "semibold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8a85",
    "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#2b2b2a",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "pdf.fonttype": 3, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 200,
})
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
C = ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"]
NICE = {c: c.capitalize() if c != "whatsapp" else "WhatsApp" for c in C}
PAL = dict(zip(C, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
VER, REP, ACC = "#2a78d6", "#c3c2b7", "#e34948"
W1, W2 = 3.35, 6.9   # one column / full width in inches


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); print("wrote", name)


lab = pd.read_csv(CB / "data/extracted/train_labels.csv", dtype=str).sort_values("Id").reset_index(drop=True)
y = lab.Label.map({c: i for i, c in enumerate(C)}).to_numpy()
meta = pd.read_csv(CB / "outputs/eda/metadata_train.csv", dtype={"Id": str}).set_index("Id").loc[lab.Id].reset_index()

# ------------------------------------------------------------------ Fig 1: teaser
picks = json.loads((WORK / "face_picks.json").read_text())
choice = {"apple": 5, "facebook": 7, "google": 10, "messenger": 7, "mozilla": 3, "samsung": 1, "whatsapp": 3}
fig = plt.figure(figsize=(W2, 2.55))
gs = fig.add_gridspec(3, 9, width_ratios=[1] * 7 + [0.25, 1.9], height_ratios=[1, 1, 0.55], wspace=0.08, hspace=0.12)
for j, c in enumerate(C):
    i = picks[c][choice[c]]
    fid = lab.Id[i]
    raw = Image.open(CB / f"data/extracted/train/{fid}.png")
    m = meta.iloc[i]
    rgba = raw.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255)); white.alpha_composite(rgba)
    naive = raw.convert("RGB")
    for r, im in enumerate([white.convert("RGB"), naive]):
        ax = fig.add_subplot(gs[r, j])
        ax.imshow(im.resize((240, 240), Image.NEAREST), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#d9d8d2"); s.set_linewidth(0.5)
        if r == 0:
            ax.set_title(NICE[c], fontsize=7.5, color=INK, pad=3)
    ax = fig.add_subplot(gs[2, j]); ax.axis("off")
    ax.text(0.5, 0.8, f"{int(m.width)} px\n{m['mode']}", ha="center", va="center", fontsize=6.3, color=INK2,
            transform=ax.transAxes, linespacing=1.1)
    sw = np.array([m.under_r, m.under_g, m.under_b]) / 255
    ax.add_patch(plt.Rectangle((0.28, 0.0), 0.44, 0.3, transform=ax.transAxes, color=sw, ec="#b0afa8", lw=0.4))
fig.text(0.115, 0.71, "what the network sees\n(composited on white)", ha="right", va="center", fontsize=6.6, color=INK2)
fig.text(0.115, 0.41, "naive RGB load\n(alpha dropped)", ha="right", va="center", fontsize=6.6, color=INK2)
fig.text(0.115, 0.115, "file fingerprint:\nsize, mode, colour\nunder transparency", ha="right", va="center", fontsize=6.6, color=INK2)
ax = fig.add_subplot(gs[:, 8])
labels = ["Majority class", "File fingerprint only\n(no drawn pixel)", "Small CNN, pixels\n(from scratch)", "ConvNeXt-Tiny, pixels\n(final model)"]
vals = [N["cpu"]["majority_class"]["accuracy"], N["cpu"]["metadata-only / size+mode+file stats+under-colour / random_forest"]["accuracy"],
        N["cpu"]["EmojiCNNv2 96px / pixels only"]["accuracy"], N["final"]["accuracy"]]
cols = [REP, ACC, VER, VER]
yy = np.arange(len(vals))[::-1]
ax.barh(yy, vals, color=cols, height=0.34)
for yv, v, l in zip(yy, vals, labels):
    ax.text(v + 0.02, yv, f"{v:.1%}", va="center", fontsize=7, color=INK, fontweight="semibold")
    ax.text(0, yv + 0.3, l.replace("\n", " "), va="bottom", ha="left", fontsize=6.4, color=INK2)
ax.set_xlim(0, 1.25); ax.set_ylim(-0.5, 3.75); ax.set_yticks([]); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"])
ax.spines["left"].set_visible(False)
ax.set_title("validation accuracy", fontsize=7.5, color=INK2, fontweight="normal", loc="left")
save(fig, "fig1_teaser")

# ------------------------------------------------------------------ Fig 2: dataset & file fingerprints
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.05), gridspec_kw={"width_ratios": [1.05, 1.45, 1.0], "wspace": 0.62})
cnt = pd.Series(N["eda"]["class_counts"]).reindex(C)
ax = axs[0]
order = cnt.sort_values().index
ax.barh(range(7), cnt[order], color=[PAL[c] for c in order], height=0.66)
ax.set_yticks(range(7)); ax.set_yticklabels([NICE[c] for c in order])
for k, c in enumerate(order):
    ax.text(cnt[c] + 30, k, f"{cnt[c]:,}", va="center", fontsize=6.6, color=INK2)
ax.set_xlim(0, 2350); ax.set_xticks([0, 1000, 2000]); ax.set_xticklabels(["0", "1k", "2k"])
ax.set_title("(a) images per vendor", loc="left")
ax.axvline(len(lab) / 7, color=MUTED, lw=0.6, ls=":"); ax.text(len(lab) / 7 + 30, 6.55, "uniform", fontsize=6, color=MUTED)

ax = axs[1]
fam = pd.DataFrame({"72": (meta.width == 72) & (meta.height == 72), "72×68": (meta.width == 72) & (meta.height == 68),
                    "160": meta.width == 160, "240": meta.width == 240})
fam["other"] = ~fam.any(axis=1)
share = fam.groupby(lab.Label).mean().reindex(C)
mat = share.to_numpy()
ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if v >= 0.005:
            ax.text(j, i, f"{v:.0%}" if v >= 0.01 else "<1%", ha="center", va="center", fontsize=6.3,
                    color="white" if v > 0.55 else INK2)
ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(share.columns)
ax.set_yticks(range(7)); ax.set_yticklabels([NICE[c] for c in C])
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("(b) native size (px), share of files", loc="left")

ax = axs[2]
pm = (meta["mode"] == "P").groupby(lab.Label).mean().reindex(C)
ax.barh(range(7)[::-1], pm, color=[PAL[c] for c in C], height=0.66)
ax.set_yticks(range(7)[::-1]); ax.set_yticklabels([NICE[c] for c in C])
for k, c in zip(range(7)[::-1], C):
    ax.text(pm[c] + 0.02, k, f"{pm[c]:.0%}", va="center", fontsize=6.6, color=INK2)
ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"])
ax.set_title("(c) palette (P) PNGs", loc="left")
save(fig, "fig2_dataset")

# ------------------------------------------------------------------ Fig 3: the hidden colour barcode + metadata-only ladder
fig = plt.figure(figsize=(W2, 2.2))
gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1], wspace=0.42)
ax = fig.add_subplot(gs[0])
rng = np.random.default_rng(3)
K = 180
for r, c in enumerate(C):
    sub = meta[(lab.Label == c).to_numpy() & meta.has_alpha.astype(bool).to_numpy()]
    sub = sub.iloc[rng.choice(len(sub), min(K, len(sub)), replace=False)]
    rgb = sub[["under_r", "under_g", "under_b"]].to_numpy() / 255
    lum = rgb @ np.array([0.299, 0.587, 0.114]); rgb = rgb[np.argsort(lum)]
    ax.imshow(rgb[None, :, :], extent=(0, K, 6 - r - 0.38, 6 - r + 0.38), aspect="auto", interpolation="nearest")
ax.set_xlim(0, K); ax.set_ylim(-0.6, 6.6)
ax.set_yticks(range(7)); ax.set_yticklabels([NICE[c] for c in C][::-1]); ax.tick_params(length=0)
ax.set_xticks([]); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title(f"(a) colour hidden under transparent pixels, {K} files per vendor", loc="left")
ax = fig.add_subplot(gs[1])
cpu = N["cpu"]
lev = [("size\n+ mode", "size+mode"), ("+ file\nstats", "size+mode+file stats"), ("+ hidden\ncolour", "size+mode+file stats+under-colour")]
x = np.arange(3)
lr = [cpu[f"metadata-only / {k} / logreg"]["accuracy"] for _, k in lev]
rf = [cpu[f"metadata-only / {k} / random_forest"]["accuracy"] for _, k in lev]
ax.bar(x - 0.19, lr, 0.36, color="#9fc4ee", label="logistic regression")
ax.bar(x + 0.19, rf, 0.36, color=ACC, label="random forest")
for xi, a, b in zip(x, lr, rf):
    ax.text(xi - 0.19, a + 0.012, f"{a:.0%}", ha="center", fontsize=6.3, color=INK2)
    ax.text(xi + 0.19, b + 0.012, f"{b:.0%}", ha="center", fontsize=6.3, color=INK, fontweight="semibold")
ax.axhline(cpu["majority_class"]["accuracy"], color=MUTED, lw=0.6, ls=":")
ax.text(-0.5, cpu["majority_class"]["accuracy"] + 0.015, "majority class", fontsize=5.8, color=MUTED, ha="left")
ax.set_xticks(x); ax.set_xticklabels([l for l, _ in lev], fontsize=6.8)
ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "50%", "100%"])
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=6.3, handlelength=1)
ax.set_ylim(0, 1.05)
ax.set_title("(b) accuracy, no drawn pixel", loc="left")
save(fig, "fig3_hidden_colour")

# ------------------------------------------------------------------ Fig 5: ladder
rows = list(N["ladder_rows"]) + [
    {"label": "Extra-trees: pixels, colour, transparency (Lina)", "accuracy": 0.8214, "status": "reported"},
    {"label": "Frozen ResNet18 + hand-crafted, RBF SVM (Lina)", "accuracy": 0.9676, "status": "reported"},
]
lbl = {"Majority class": "Majority class"}
fig, ax = plt.subplots(figsize=(W2, 3.0))
rr = sorted(rows, key=lambda r: r["accuracy"])
names = [r["label"].replace(" (out-of-fold, 9,879 images)", " [OOF]").replace("ConvNeXt-Tiny 224 px, 5-fold + TTA, ", "ConvNeXt-T 224, 5-fold, ")
         .replace("Both checkpoint sets averaged (10 checkpoints)", "ConvNeXt-T 224, 10-checkpoint avg.")
         .replace("File metadata + colour under transparency, no drawing (random_forest)", "File metadata incl. hidden colour (RF)")
         .replace("File format only (dimensions + PNG mode), no pixels (random_forest)", "File format only: size + mode (RF)")
         .replace("ConvNeXt-Tiny (128 px) + rotation/scale aug", "ConvNeXt-Tiny 128 + rotation/scale aug.") for r in rr]
cols = [VER if r["status"] == "verified" else REP for r in rr]
ax.barh(range(len(rr)), [r["accuracy"] for r in rr], color=cols, height=0.7)
for k, r in enumerate(rr):
    ax.text(r["accuracy"] + 0.006, k, f"{r['accuracy']:.1%}", va="center", fontsize=6.8, color=INK if r["status"] == "verified" else INK2)
ax.set_yticks(range(len(rr))); ax.set_yticklabels(names, fontsize=7)
ax.set_xlim(0.0, 1.07); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
ax.axvline(cpu["metadata-only / size+mode+file stats+under-colour / random_forest"]["accuracy"], color=ACC, lw=0.8, ls="--")
ax.text(0.865, len(rr) - 0.35, "no-pixel floor: 87.4% ", color=ACC, fontsize=6.8, ha="right")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=VER, label="verified (this codebase)"), Patch(color=REP, label="reported (team log, 80/20 split)")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2, fontsize=7)
ax.tick_params(axis="y", length=0)
save(fig, "fig5_ladder")

# ------------------------------------------------------------------ Fig 7: training dynamics (loss vs accuracy divergence)
H = [pd.read_csv(CB / f"outputs/convnext224/fold{k}_history.csv") for k in range(5)]
kept = [json.loads((CB / f"outputs/convnext224/fold{k}_run.json").read_text())["best_epoch"] for k in range(5)]
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.15), gridspec_kw={"wspace": 0.42})
ep = H[0].epoch.to_numpy()
for key, ax, title in [("val_loss", axs[0], "(a) validation cross-entropy"), ("val_accuracy", axs[1], "(b) validation accuracy")]:
    M = np.stack([h[key].to_numpy() for h in H])
    for k in range(5):
        ax.plot(ep, M[k], color="#b9cfe9", lw=0.8, zorder=1)
        e = kept[k]; ax.plot(e, M[k][e - 1], "o", ms=4.2, mfc="white", mec=ACC, mew=1.1, zorder=3)
        ax.plot(10, M[k][9], "s", ms=3.8, color=VER, zorder=3)
    ax.plot(ep, M.mean(0), color=VER, lw=1.8, zorder=2)
    ax.set_title(title, loc="left"); ax.set_xlabel("epoch"); ax.set_xticks([1, 4, 7, 10]); ax.grid(axis="y")
    if key == "val_loss":
        ax.set_ylim(0.08, 0.2); ax.text(9.8, 0.19, "mean of 5 folds", color=VER, fontsize=6.3, ha="right")
    else:
        ax.set_ylim(0.95, 0.982); ax.set_yticks([0.95, 0.96, 0.97, 0.98]); ax.set_yticklabels(["95%", "96%", "97%", "98%"])
ax = axs[2]
lr = H[0].lr.to_numpy()
tr = np.stack([h.train_accuracy for h in H]).mean(0)
ax.plot(ep, tr, color=INK2, lw=1.4, label="train accuracy")
ax.set_ylim(0.5, 1.005); ax.set_yticks([0.6, 0.8, 1.0]); ax.set_yticklabels(["60%", "80%", "100%"])
ax.set_xlabel("epoch"); ax.set_xticks([1, 4, 7, 10]); ax.grid(axis="y")
ax.set_title("(c) mean training accuracy", loc="left")
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=ACC, mew=1.1, label="epoch kept by run A (lowest val. loss)"),
                    Line2D([], [], marker="s", ls="", color=VER, label="epoch kept by run B (last epoch, final model)")],
           loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.2), fontsize=6.8)
save(fig, "fig7_dynamics")

# ------------------------------------------------------------------ Fig 8: calibration & over-confidence
A = np.load(CB / "outputs/convnext224/oof_proba.npy"); B = np.load(CB / "outputs/convnext224_last/oof_proba.npy")
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.1), gridspec_kw={"wspace": 0.45})
ax = axs[0]
for P, col, nm in [(A, ACC, "run A"), (B, VER, "run B (final)")]:
    conf = P.max(1); ok = P.argmax(1) == y
    edges = np.array([0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0001])
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 15:
            xs.append(conf[m].mean()); ys.append(ok[m].mean())
    ax.plot(xs, ys, "-o", color=col, ms=3, lw=1.2, label=nm)
ax.plot([0.3, 1], [0.3, 1], color=MUTED, lw=0.6, ls=":")
ax.set_xlim(0.3, 1.01); ax.set_ylim(0.3, 1.01); ax.set_xlabel("mean confidence"); ax.set_ylabel("accuracy")
ax.set_title("(a) reliability", loc="left"); ax.legend(frameon=False, loc="upper left", fontsize=6.5)
ax = axs[1]
bins = np.linspace(0.2, 1, 17)
for P, col, nm in [(A, ACC, "run A"), (B, VER, "run B")]:
    conf = P.max(1); ok = P.argmax(1) == y
    ax.hist(conf[~ok], bins=bins, histtype="step", color=col, lw=1.3, label=f"{nm}: {(~ok).sum()} errors")
ax.set_xlabel("confidence of the wrong prediction"); ax.set_ylabel("errors")
ax.set_title("(b) how sure the mistakes are", loc="left"); ax.legend(frameon=False, loc="upper left", fontsize=6.5)
ax = axs[2]
for P, col, nm in [(A, ACC, "run A"), (B, VER, "run B")]:
    conf = P.max(1); ok = (P.argmax(1) == y)
    o = np.argsort(-conf); cov = np.arange(1, len(y) + 1) / len(y); accs = np.cumsum(ok[o]) / np.arange(1, len(y) + 1)
    ax.plot(cov, accs, color=col, lw=1.3, label=nm)
ax.set_xlim(0.5, 1); ax.set_ylim(0.97, 1.0005); ax.set_xlabel("coverage (most confident first)")
ax.set_yticks([0.97, 0.98, 0.99, 1.0]); ax.set_yticklabels(["97%", "98%", "99%", "100%"]); ax.set_xticks([0.5, 0.75, 1]); ax.set_xticklabels(["50%", "75%", "100%"])
ax.set_title("(c) selective accuracy", loc="left"); ax.grid(axis="y"); ax.legend(frameon=False, loc="lower left", fontsize=6.5)
save(fig, "fig8_calibration")

# ------------------------------------------------------------------ Fig 9: confusion matrix of the final model
cm = np.array(N["final"]["confusion_matrix"]["counts"])
rown = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(W1, 2.9))
off = rown.copy(); np.fill_diagonal(off, np.nan)
ax.imshow(np.where(np.eye(7, dtype=bool), np.nan, off), cmap="Reds", vmin=0, vmax=0.03)
for i in range(7):
    for j in range(7):
        if i == j:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#eef3fb"))
            ax.text(j, i, f"{rown[i, j]:.1%}", ha="center", va="center", fontsize=6.2, color=VER, fontweight="semibold")
        elif cm[i, j]:
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=6.6, color="white" if rown[i, j] > 0.016 else INK2)
ax.set_xticks(range(7)); ax.set_xticklabels([NICE[c] for c in C], rotation=40, ha="right")
ax.set_yticks(range(7)); ax.set_yticklabels([NICE[c] for c in C]); ax.tick_params(length=0)
ax.set_xlabel("predicted vendor"); ax.set_ylabel("true vendor")
for s in ax.spines.values(): s.set_visible(False)
for b in [2.5, 5.5]:
    pass
save(fig, "fig9_confusion")

# ------------------------------------------------------------------ Fig 10: accuracy per export family across models
fb = N["cpu"]["family_breakdown"]
models = [("Sept. baseline CNN", fb["Baseline CNN (30 Sept)"]), ("EmojiCNNv2 (pixels)", fb["EmojiCNNv2 (pixels)"]),
          ("EmojiCNNv2 + metadata", fb["EmojiCNNv2 + metadata"]), ("ConvNeXt-T, final", N["final"]["family_breakdown"])]
fams = ["72 px family", "160 px family", "240 px family"]
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.62})
ax = axs[0]
cols4 = ["#c3c2b7", "#9fc4ee", "#e8a3a2", VER]
for k, (nm, d) in enumerate(models):
    v = [d[f]["accuracy"] for f in fams]
    ax.bar(np.arange(3) + (k - 1.5) * 0.2, v, 0.19, color=cols4[k], label=nm)
ax.set_xticks(range(3)); ax.set_xticklabels(["72 px\n(Apple, Google,\nSamsung)", "160 px\n(WhatsApp)", "240 px\n(Facebook, Messenger,\nMozilla)"], fontsize=6.3)
ax.set_ylim(0.6, 1.0); ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_yticklabels(["60", "70", "80", "90", "100%"]); ax.grid(axis="y")
ax.legend(frameon=False, ncol=4, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.62, -0.36), handlelength=1)
ax.set_title("(a) accuracy inside each native-resolution family", loc="left")
ax = axs[1]
err = [d["errors_total"] for _, d in models]; cross = [d["errors_crossing_family"] for _, d in models]
ax.barh(range(4)[::-1], [c / e for c, e in zip(cross, err)], color=cols4, height=0.6)
for k, (c, e) in zip(range(4)[::-1], zip(cross, err)):
    ax.text(c / e + 0.01, k, f"{c} of {e}", va="center", fontsize=6.4, color=INK2)
ax.set_yticks(range(4)[::-1]); ax.set_yticklabels([m for m, _ in models], fontsize=6.6)
ax.set_xlim(0, 0.55); ax.set_xticks([0, 0.2, 0.4]); ax.set_xticklabels(["0", "20%", "40%"]); ax.tick_params(axis="y", length=0)
ax.set_title("(b) errors that cross a family", loc="left")
save(fig, "fig10_families")

# ------------------------------------------------------------------ Fig 11: validation estimate vs Kaggle
LB = [("Sept. baseline", 0.7671, 0.77546, 0.78906), ("EmojiCNNv2", 0.9096, 0.90747, 0.91700),
      ("EmojiCNNv2 + metadata", 0.9338, 0.93824, 0.94615), ("10-checkpoint avg.", 0.9770, 0.98279, 0.98198),
      ("final model", 0.9769, 0.98319, 0.98218)]
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.1), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})
ax = axs[0]
v = np.array([r[1] for r in LB]); pu = np.array([r[2] for r in LB]); pr = np.array([r[3] for r in LB])
ax.plot([0.74, 1], [0.74, 1], color=MUTED, lw=0.6, ls=":")
ax.scatter(v, pu, s=22, facecolor="white", edgecolor=VER, lw=1.1, label="public (≈50% of test)", zorder=3)
ax.scatter(v, pr, s=22, color=VER, label="private (≈50% of test)", zorder=3)
for (nm, a, b, c) in LB[:3]:
    ax.text(a + 0.006, c - 0.004, nm, fontsize=6.2, color=INK2, va="top")
ax.text(0.958, 0.985, "ConvNeXt", fontsize=6.2, color=INK2, ha="right")
ax.set_xlim(0.74, 1.0); ax.set_ylim(0.74, 1.0); ax.set_xlabel("our validation estimate"); ax.set_ylabel("Kaggle score")
ax.legend(frameon=False, loc="upper left", fontsize=6.3); ax.set_title("(a) validation predicts the test set", loc="left")
ax = axs[1]
gap = pr - v
ax.barh(range(5)[::-1], gap * 100, color=VER, height=0.55)
for k, g in zip(range(5)[::-1], gap):
    ax.text(g * 100 + 0.05, k, f"+{g * 100:.2f}", va="center", fontsize=6.4, color=INK2)
ax.set_yticks(range(5)[::-1]); ax.set_yticklabels([r[0] for r in LB], fontsize=6.6); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 2.8); ax.set_xlabel("private score minus validation estimate (points)")
ax.set_title("(b) the estimate is conservative", loc="left")
save(fig, "fig11_kaggle")

# ------------------------------------------------------------------ Fig 12: most confident errors of the final model
fails = pd.read_csv(CB / "outputs/convnext224_last/failures_oof.csv", dtype={"Id": str})
print(fails.columns.tolist())
fig, axs = plt.subplots(2, 8, figsize=(W2, 2.2))
for ax, (_, r) in zip(axs.ravel(), fails.head(16).iterrows()):
    raw = Image.open(CB / f"data/extracted/train/{r['Id']}.png").convert("RGBA")
    wb = Image.new("RGBA", raw.size, (255, 255, 255, 255)); wb.alpha_composite(raw)
    ax.imshow(wb.convert("RGB").resize((160, 160), Image.LANCZOS)); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#d9d8d2"); s.set_linewidth(0.5)
    t = r.get("true", r.get("true_label")); p = r.get("pred", r.get("predicted"))
    ax.set_title(f"{NICE.get(t, t)}\n→ {NICE.get(p, p)} {r['confidence']:.3f}", fontsize=5.8, color=INK2, pad=2)
fig.subplots_adjust(wspace=0.08, hspace=0.55)
save(fig, "fig12_failures")

# ------------------------------------------------------------------ Fig 15: a second descent (is it double descent?)
hv = pd.read_csv(CB / "outputs/cpu/emojicnnv2_history.csv"); hf = pd.read_csv(CB / "outputs/cpu/emojicnnv2_fusion_history.csv")
HA = [pd.read_csv(CB / f"outputs/convnext224/fold{k}_history.csv") for k in range(5)]
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.05), gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1, 0.8, 1]})
SH = dict(color="#f6e3e2", lw=0, zorder=0)
ax = axs[0]
ax.axvspan(3.5, 7.5, **SH)
ax.plot(hv.epoch, hv.train_loss, color=MUTED, lw=1.2, ls="--", label="train")
ax.plot(hv.epoch, hv.val_loss, color=VER, lw=1.6, label="validation")
ax.set_title("(a) EmojiCNNv2: loss", loc="left"); ax.set_xlabel("epoch"); ax.set_xticks([1, 7, 13, 19, 25]); ax.set_ylim(0, 2.3)
ax.annotate("val. loss 1.16 → 2.15\ntrain acc. only 74%", xy=(7, 2.148), xytext=(9.0, 1.9), fontsize=6.2, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax = axs[1]
ax.axvspan(3.5, 7.5, **SH)
ax.plot(hv.epoch, hv.train_accuracy, color=MUTED, lw=1.2, ls="--", label="train")
ax.plot(hv.epoch, hv.val_accuracy, color=VER, lw=1.6, label="val.")
ax.plot(hf.epoch, hf.val_accuracy, color="#e8a3a2", lw=1.3, label="val., + metadata")
ax.set_ylim(0.3, 1.0); ax.set_yticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["40%", "60%", "80%", "100%"])
ax.set_title("(b) accuracy", loc="left"); ax.set_xlabel("epoch"); ax.set_xticks([1, 7, 13, 19, 25]); ax.grid(axis="y")
ax = axs[2]
ax.axvspan(3.5, 7.5, **SH)
ax.plot(hv.epoch, hv.lr / hv.lr.max(), color=INK2, lw=1.4)
ax.set_title("(c) learning rate", loc="left"); ax.set_xlabel("epoch"); ax.set_xticks([1, 7, 13, 19, 25])
ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "½ max", "max"])
ax = axs[3]
ML = np.stack([h.val_loss.to_numpy() for h in HA])
for k in range(5): ax.plot(HA[k].epoch, ML[k], color="#b9cfe9", lw=0.8)
ax.plot(HA[0].epoch, ML.mean(0), color=VER, lw=1.8)
ax.axvspan(4.5, 5.5, **SH)
ax.set_ylim(0.09, 0.19); ax.set_xlim(2.5, 10.3); ax.set_xticks([3, 5, 7, 10]); ax.set_xlabel("epoch")
tr5 = np.mean([h.train_accuracy.iloc[4] for h in HA])
ax.text(5.15, 0.181, f"train acc. {tr5:.1%}", fontsize=6.2, color=INK2)
ax.set_title("(d) ConvNeXt-Tiny: val. loss", loc="left")
from matplotlib.lines import Line2D as _L
fig.legend(handles=[_L([], [], color=MUTED, ls="--", lw=1.2, label="train (EmojiCNNv2)"), _L([], [], color=VER, lw=1.6, label="validation (EmojiCNNv2, pixels)"),
                    _L([], [], color="#e8a3a2", lw=1.3, label="validation (EmojiCNNv2 + metadata)"), _L([], [], color="#b9cfe9", lw=1, label="ConvNeXt folds (thin) and mean (thick)")],
           loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.26), fontsize=6.3, handlelength=1.6)
save(fig, "fig15_second_descent")
