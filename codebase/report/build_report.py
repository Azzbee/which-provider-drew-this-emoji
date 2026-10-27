"""Render report/template.html with report/numbers.json into Report_The_Winners.html and .pdf (Chromium)."""
from __future__ import annotations

import base64
import json
import sys
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N = json.loads((HERE / "numbers.json").read_text())
import os
OUTS = Path(os.environ.get("EMOJI_OUTPUTS", ROOT / "outputs"))
GPU_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTS / "convnext224"


def img(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * float(x):.{d}f}%"


def num(x, d=3):
    return "n/a" if x is None else f"{float(x):.{d}f}"


env = Environment(loader=FileSystemLoader(str(HERE)), autoescape=select_autoescape(["html"]))
env.filters["pct"] = pct
env.filters["num"] = num
tpl = env.get_template("template.html")

F = {
    "class_distribution": img(OUTS / "eda/figures/class_distribution.png"),
    "samples": img(OUTS / "eda/figures/samples_per_provider.png"),
    "native_size": img(OUTS / "eda/figures/native_size_by_provider.png"),
    "native_mode": img(OUTS / "eda/figures/native_mode_by_provider.png"),
    "alpha": img(OUTS / "eda/figures/alpha_naive_vs_white.png"),
    "cm_meta": img(OUTS / "cpu/figures/cm_metadata_only.png"),
    "cm_baseline": img(OUTS / "cpu/figures/cm_baseline_sept30_norm.png"),
    "curves_v2": img(OUTS / "cpu/figures/curves_emojicnnv2.png"),
    "cm_v2": img(OUTS / "cpu/figures/cm_emojicnnv2_norm.png"),
    "cm_fusion": img(OUTS / "cpu/figures/cm_emojicnnv2_fusion.png"),
    "per_class_f1": img(OUTS / "cpu/figures/per_class_f1_cpu_models.png"),
    "failures_v2": img(OUTS / "cpu/figures/failures_emojicnnv2.png"),
    "gradcam": img(OUTS / "cpu/figures/gradcam_emojicnnv2.png"),
    "ladder": img(HERE / "figures/ladder.png"),
    "cm_oof": img(GPU_DIR / "figures/cm_oof.png"),
    "cm_oof_norm": img(GPU_DIR / "figures/cm_oof_norm.png"),
    "failures_oof": img(GPU_DIR / "figures/failures_oof.png"),
    "curves_fold0": img(GPU_DIR / "figures/curves_fold0.png"),
    "cm_final": img(HERE / "figures/cm_final_oof.png"),
    "cm_final_norm": img(HERE / "figures/cm_final_oof_norm.png"),
    "failures_final": img(HERE / "figures/failures_final.png"),
    "curves_runs": img(HERE / "figures/curves_convnext_runs.png"),
}
cpu = N.get("cpu") or {}
_levels = ["size+mode", "size+mode+file stats", "size+mode+file stats+under-colour"]
meta_keys = sorted([k for k in cpu if k.startswith("metadata-only")],
                   key=lambda k: (_levels.index(k.split("/")[1].strip()) if k.split("/")[1].strip() in _levels else 9, k))
best_meta = max((cpu[k] for k in meta_keys), key=lambda v: v["accuracy"]) if meta_keys else None
fmt_keys = sorted([k for k in meta_keys if k.split("/")[1].strip() == "size+mode"], key=lambda k: cpu[k]["accuracy"])
meta_format = cpu[fmt_keys[-1]] if fmt_keys else None              # pure container format, nothing from the drawing
best_key = max(meta_keys, key=lambda k: cpu[k]["accuracy"]) if meta_keys else None
html = tpl.render(N=N, F=F, today=date.today().strftime("%d %B %Y"), meta_keys=meta_keys, best_meta=best_meta,
                  best_meta_key=best_key, meta_format=meta_format, meta_format_key=(fmt_keys[-1] if fmt_keys else None))
out_html = HERE / "Report_The_Winners.html"
out_html.write_text(html, encoding="utf-8")
print("wrote", out_html)

try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page()
        pg.goto(out_html.resolve().as_uri())
        pg.wait_for_timeout(500)
        pg.pdf(path=str(HERE / "Report_The_Winners.pdf"), format="A4", print_background=True,
               margin={"top": "18mm", "bottom": "18mm", "left": "17mm", "right": "17mm"},
               display_header_footer=True,
               header_template="<div></div>",
               footer_template="<div style='font-size:8px;color:#777;width:100%;text-align:center;font-family:Helvetica,Arial'>"
                               "Emoji provider classification, IDSI51006, team The Winners, page <span class='pageNumber'></span> / <span class='totalPages'></span></div>")
        b.close()
    print("wrote", HERE / "Report_The_Winners.pdf")
except Exception as exc:  # pragma: no cover
    print("PDF step skipped:", exc)
    print("Open the HTML in a browser and print to PDF, or pip install playwright && playwright install chromium")
