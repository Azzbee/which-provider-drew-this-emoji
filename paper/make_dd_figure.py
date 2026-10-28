"""Figure and numbers for the 2,000-epoch double-descent experiment (logs from the RunPod run)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
LOGS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "members/adam-zeraiki/double-descent-2000-epochs/logs"
OUT = HERE / "figs"

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
INK2 = "#52514e"
NOISE_COL = {0: "#9cc3ee", 10: "#2a78d6", 20: "#0e3a73"}     # one hue, light to dark = more noise
W2 = 6.9


def load(name):
    rows = [json.loads(l) for l in open(LOGS / f"{name}.jsonl")]
    head, rows = rows[0], rows[1:]
    g = lambda k: np.array([r[k] if r[k] is not None else np.nan for r in rows], float)
    return head, {k: g(k) for k in rows[0]}


def smooth(x, k=9):
    if len(x) < k: return x
    p = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(p, np.ones(k) / k, mode="valid")


summary = {}
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.15), gridspec_kw={"wspace": 0.36})
for n in (0, 10, 20):
    head, d = load(f"w64_n{n}_s0")
    ep = d["epoch"]; dev_err = 100 * (1 - d["dev_acc"]); tr_err = 100 * (1 - d["train_acc"])
    ax[0].plot(ep, dev_err, color=NOISE_COL[n], lw=0.35, alpha=0.35)
    ax[0].plot(ep, smooth(dev_err), color=NOISE_COL[n], lw=1.3, label=f"{n}% of labels wrong")
    ax[1].plot(ep, tr_err, color=NOISE_COL[n], lw=1.3)
    sm = smooth(dev_err)
    i_min = int(np.argmin(sm[:max(10, len(sm))]))
    # first epoch at which training error on the (noisy) labels drops below 1%
    interp = int(ep[np.argmax(tr_err < 1.0)]) if (tr_err < 1.0).any() else None
    after = sm[i_min:]
    summary[f"w64_n{n}"] = {
        "epochs_run": int(ep[-1]), "params": head["params"],
        "best_dev_acc": float(100 * d["dev_acc"].max()), "best_epoch": int(ep[np.argmax(d["dev_acc"])]),
        "smoothed_min_dev_err": float(sm.min()), "smoothed_min_epoch": int(ep[np.argmin(sm)]),
        "final_dev_acc_mean_last50": float(100 * d["dev_acc"][-50:].mean()),
        "final_train_acc_mean_last50": float(100 * d["train_acc"][-50:].mean()),
        "epoch_train_err_below_1pct": interp,
        "smoothed_peak_dev_err_after_min": float(after.max()), "smoothed_peak_epoch_after_min": int(ep[i_min + int(np.argmax(after))]),
        "memorised_corrupted_last50": None if np.isnan(d["memorised_corrupted"]).all() else float(100 * np.nanmean(d["memorised_corrupted"][-50:])),
    }
for a in ax[:2]:
    a.set_xscale("log"); a.set_xlabel("epoch (log scale)"); a.grid(True, which="major")
ax[0].set_ylabel("development error (%)"); ax[0].set_title("(a) Held-out error, width 64", loc="left")
ax[1].set_ylabel("training error on its labels (%)"); ax[1].set_title("(b) Training error, width 64", loc="left")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="upper center", bbox_to_anchor=(0.355, -0.09), ncol=3)

widths = [4, 8, 16, 32, 64]
fin, best, trn = [], [], []
for w in widths:
    head, d = load(f"w{w}_n20_s0")
    fin.append(100 * (1 - d["dev_acc"][-50:].mean())); best.append(100 * (1 - d["dev_acc"].max()))
    trn.append(100 * (1 - d["train_acc"][-50:].mean()))
    summary[f"width{w}_noise20"] = {"params": head["params"], "epochs_run": int(d["epoch"][-1]),
                            "final_dev_err_last50": fin[-1], "best_dev_err": best[-1], "final_train_err_last50": trn[-1]}
ax[2].plot(widths, fin, "-o", color=NOISE_COL[20], ms=4, lw=1.3, label="dev, end of training")
ax[2].plot(widths, best, "--o", color=NOISE_COL[20], ms=3.5, lw=1, mfc="white", label="dev, best epoch")
ax[2].plot(widths, trn, "-s", color="#8a8a85", ms=3.5, lw=1, label="train (noisy labels)")
ax[2].set_xscale("log", base=2); ax[2].set_xticks(widths); ax[2].set_xticklabels(widths)
ax[2].set_xlabel("width k (channels of the first stage)"); ax[2].set_ylabel("error (%)")
ax[2].set_title("(c) Model size, 20% noisy labels", loc="left"); ax[2].grid(True); ax[2].legend(frameon=False, fontsize=6.8)
fig.savefig(OUT / "fig_dd2000.pdf")
json.dump(summary, open(OUT.parent / "work" / "dd2000_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
