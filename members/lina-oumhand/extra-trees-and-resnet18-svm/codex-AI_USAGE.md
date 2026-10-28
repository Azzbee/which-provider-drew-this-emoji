# AI assistance disclosure

OpenAI Codex was used to inspect and replace the initial eight-cell exploration notebook, implement image loading, the Extra Trees baseline, the ResNet18 training/validation/refit workflow, CSV export, and setup documentation. It also ran the local CPU baseline and checked the dataset and code. At that initial stage, neural network training and Kaggle submission were not performed by the assistant. The improvement work below later extracted pretrained neural-network features and trained SVM classifiers; it did not fine-tune a neural network or submit to Kaggle.

## User prompt

The user supplied the course assignment, including the task of predicting emoji providers, Colab/Kaggle submission requirements, explainability and reproducibility expectations, and the requirement to disclose AI prompts and write the report independently. The action request was:

> With these informations can you complete all the tabs so that I can make a submission in the kaggle competition and have a good rate

This is the action prompt verbatim; the preceding course brief is summarized above. The follow-up prompt for the improvement is recorded below. Assistant tool calls are implementation steps, not additional student prompts.

## Team responsibilities

Review and understand the generated code. Record any subsequent AI prompts here, including debugging and experiment changes. Replace no measured results with unverified claims. Write the report independently and cite the assistance, specifying what was accepted, changed, and verified by the team.

## Improvement follow-up prompt

> The kaggle score for this submission is [0.81838](https://www.kaggle.com/competitions/2-computer-vision-2026-b-sc-aidams-final-proj/submissions#). Improve it please

Codex designed image style features (color/opacity histograms, gradients, local binary texture, spatial statistics), extracted channel statistics from a frozen ImageNet ResNet18, compared SVM settings on the existing seed-42 split, checked the selected model with seed 123, and refit it on all labeled images to create a new CSV. It installed the dependencies in the existing virtual environment, wrote reusable scripts, updated the main notebook and instructions, and preserved the earlier notebook and submission. Training and standardization use only the training fold during validation. Test features are computed without test labels. The main notebook was syntax-checked; the underlying improved training/export script was executed locally. The earlier optional ResNet fine-tuning notebook remains unexecuted.

Measured improvement: 82.14% → 96.76% validation accuracy on seed 42; the second split scored 97.42%. The new Kaggle score is not known. These figures must not be reported as leaderboard performance or as an independent held-out test result after model selection. The team should explain the hand-designed feature families and the frozen-feature/SVM pipeline in its own words and record future prompts and experiments here.
