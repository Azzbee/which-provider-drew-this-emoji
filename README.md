# Which provider drew this emoji?

Team **The Winners** (Adam Zeraiki, Lina Oumhand, Charles Montluc-Giannopoulou, Erdem Sanver), IDSI51006 Computer Vision, BSc AIDAMS, ESSEC x CentraleSupélec, Winter 2026.

We classify emoji images by the vendor that drew them (Apple, Facebook, Google, Messenger, Mozilla, Samsung, WhatsApp) for the course's Kaggle competition. This repository holds the research paper, the code that produced every verified number in it, and one folder per team member with the methods each of us tried.

**Read the paper:** [`paper/Paper_The_Winners.pdf`](paper/Paper_The_Winners.pdf)

## Results

| | Score |
|---|---|
| Final model | ConvNeXt-Tiny, ImageNet weights, fine-tuned at 224 px, 5 grouped folds, flip TTA, last-epoch checkpoints |
| Out-of-fold accuracy (all 9,879 training images) | **97.69%** (95% CI 97.40 to 97.98) |
| Kaggle, final model, public / private | 0.98319 / **0.98218** |
| Kaggle, team (best selected entry: Erdem's E15) | 0.98238 private, **5th of 18 teams** |

Three findings carry the paper: the PNG files leak the label (a random forest reaches 87.4% without reading a drawn pixel, so we removed that shortcut); input resolution mattered more than architecture; and keeping the last epoch of the OneCycle schedule beats keeping the lowest-validation-loss epoch by 0.40 points.

## Repository map

```
paper/                  the LaTeX paper (main.tex + parts/*.tex), bibliography, figures, figure script
  Paper_The_Winners.pdf   compiled paper
  make_figures.py         rebuilds every figure from codebase/outputs (no number typed by hand)
  parts/                  background, architecture diagrams, fusion, journey table, who-did-what
codebase/               the submitted code: src/emoji_provider, scripts, notebook, tests, saved outputs
  outputs/                probabilities, histories, metrics and submissions of every verified run
  README.md               how to rerun everything (laptop, Colab, Databricks, RunPod)
  AI_USAGE.md             every AI tool used and the prompts (required by the course)
members/                one folder per person: what they tried, scores, code or notes
  <name>/<method>/        one subfolder per method, named after what was tried
  kaggle_submissions_ledger_2026-10-18.csv   every team submission with its uploader and score
TIMELINE.md             dated history of the project, reconstructed from Kaggle and run logs
AGENTS.md               rules for teammates and AI coding agents working on this repo
```

## Build the paper

```bash
cd paper && make            # latexmk + pdflatex + bibtex
```

Or on Overleaf: upload the `paper/` folder as a new project and compile with pdfLaTeX.

## Rerun the experiments

See [`codebase/README.md`](codebase/README.md). Download the competition archive from Kaggle first and extract it into `codebase/data/extracted/`; the images are not in this repository.

## Contributing

Read [`AGENTS.md`](AGENTS.md) first. In short: put your own work in `members/<your-name>/`, check every number against a saved output before it goes into the paper, and never commit data, model weights or keys.
