# AI usage

Team The Winners: Adam Zeraiki, Lina Oumhand, Charles Montluc-Giannopoulou, Erdem Sanver.

We used Claude, ChatGPT and Codex to discuss approaches, understand methods, generate and debug code, run experiments and prepare documentation. The team made the experimental decisions, reviewed the outputs and checked the reported results against saved records.

## Adam Zeraiki

**Contribution:** Coordinated the project and proposed research directions including self-supervised fine-tuning, model fusion and decision models. Developed the shared CNN and ConvNeXt experiments, metadata diagnostics, fusion experiments, file-byte model and double-descent analyses. Selected the final model and integrated the repository and paper.

**AI assistance:** Claude helped implement and execute experiments and draft the paper, figures and tables. ChatGPT and Codex assisted with the baseline notebook, DINOv2 implementation, learned fusion and decision-model experiments.

## Erdem Sanver

**Contribution:** Designed and evaluated the E1a to E16 experiment series, comparing architectures, resolutions, preprocessing and ensembles. Produced the team's best Kaggle entry, E15, and documented its constituent models and results.

**AI assistance:** Claude Code helped implement and debug the experiments and draft documentation under Erdem's direction and review.

## Charles Montluc-Giannopoulou

**Contribution:** Explored an ImageNet-pretrained ResNet18 at 160 px with file-format features. Reviewed the notebook and corrected the paper's description to make those extra inputs explicit.

**AI assistance:** Claude helped build and review the notebook, write the method notes and check the files before adding them to the repository. The original notebook prompts were not retained.

## Lina Oumhand

**Contribution:** Explored Extra Trees and a frozen ResNet18 combined with hand-crafted features and an RBF SVM. Compared the contributions of the two feature families. Explored partial fine-tuning of DINOv2 and submitted the resulting DINOv2 prediction file.

**AI assistance:** Codex explained the methods and helped research approaches, structure notebooks, implement and debug code, run experiments and interpret results. For DINOv2, Codex assisted Lina with model setup, preprocessing, training and validation code, debugging and prediction export.
