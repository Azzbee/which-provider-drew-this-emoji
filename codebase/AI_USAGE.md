# AI usage

Team The Winners: Adam Zeraiki, Lina Oumhand, Charles Montluc-Giannopoulou, Erdem Sanver.

## How we worked with AI

We used AI assistants (Claude from Anthropic, OpenAI ChatGPT and Codex) in three ways: as a sparring partner to test and sharpen our ideas, as help to write and execute code faster, and, for members newer to deep-learning code, as a way to understand larger blocks of code before using them. The ideas, the experimental choices and the decisions were ours. Code produced with AI help was read, checked and re-run by us before any result was used, and every number in the paper was checked against the output files that produced it. Our conversations with these tools are available on request.

## Who did what

### Adam Zeraiki

**His work**
- Led the project: split the work, set the deadlines, coordinated the four of us and managed the shared Kaggle budget of five uploads a day.
- Proposed the main research directions: fine-tuning a self-supervised model (DINO), fusing several models with a transformer in the middle (inspired by TRIBE v2), testing decision models such as Jev as a final arbiter, and, when those did not beat a plain average, adding an expert that reads only the PNG file bytes so that the experts would make different kinds of mistakes.
- Brought in two outside ideas and had them tested on our data: the poisoned-cup puzzle from his interview preparation, which led to the error-correcting output codes of section 4.5, and the deep double descent paper (Nakkiran et al.), which he had read out of curiosity, tested up to 4,000 epochs and 11 network sizes.
- Asked the questions that shaped the analysis: whether we overfit or underfit, and how far the file-format shortcut alone could go.
- Reviewed every part of the code written with AI help, re-read and re-checked it, and ran experiments himself; created, paid for and managed the RunPod GPU pod, checking the price before each launch; sent the Jev requests from his own computer with his own key.
- Chose the final model and kept it pixel-only, chose the two entries selected for the private leaderboard, and decided what goes into the paper and how its story is told ("we tried this, it did not work, so we changed it").
- Set up the shared repository with one folder per member, reviewed and merged his teammates' pull requests, and checked the final paper.

**How AI helped**
- Claude: discussing ideas and their weaknesses; helping write and execute the codebase, the GPU runs, the file-byte model, the three-expert fusion, the combination rules and the double-descent replication; drafting the paper, its figures and tables from our saved outputs, which Adam reviewed and corrected.
- OpenAI ChatGPT and Codex: the September baseline notebook, and implementing the DINOv2 fine-tuning, the fusion transformer and the decision-model experiments.

### Erdem Sanver

**His work**
- Designed the E1a to E16 experiment series and its rules: one change at a time, validate locally before spending a shared upload, log every run, keep the model pixel-only.
- Reviewed the results at each step, chose what to submit, and produced the team's best entry, E15 (0.98238 on the private leaderboard), a soft vote of three ConvNeXt-Tiny models at different resolutions and resize filters.
- Documented every recipe, command and result of the series, including the bicubic versus nearest-neighbour test behind E15's resize choice, in `members/erdem-sanver/`.

**How AI helped**
- Claude Code as a coding assistant: implementing and debugging the plan he set and approved, and drafting the write-ups, which he reviewed.

### Charles Montluc-Giannopoulou

**His work**
- Explored transfer learning at a higher input resolution with an ImageNet-pretrained ResNet18 at 160 px (0.96396 on the private leaderboard).
- Reviewed his own notebook afterwards and corrected how the paper described his model, pointing out that it also used the file format and so is not a pixels-only result.

**How AI helped**
- Claude: building and later reviewing the notebook, and writing up his notes.

### Lina Oumhand

**Her work**
- Started from a simple Extra Trees baseline (82.14% validation, 0.81838 on the public leaderboard), set the goal of improving it, and moved to hand-crafted style features with frozen ResNet18 features and an RBF SVM (96.76% validation, 0.96983 private), the best result obtained without fine-tuning any network.
- Measured which part of that model matters: the hand-crafted features alone reach 96.05%, the frozen ResNet18 features alone 91.65%.
- Explored DINOv2 with partial fine-tuning and submitted the team's DINOv2 entry.

**How AI helped**
- OpenAI Codex helped Lina with explaining the notebook and methods covered in class. It also helped her research and compare different approaches for the emoji classification task. Codex assisted in building and structuring the Jupyter notebooks, especially for the Extra Trees and ResNet18 + SVM models. It helped implement the methods, write and improve the code, fix errors, debug issues, and run the experiments. It also helped Lina explore DINOv2 as another approach for the emoji classification task. It explained how the pretrained Vision Transformer works and how partial fine-tuning could be used to adapt it to the dataset. Codex assisted with setting up the model, adapting the image preprocessing, implementing the training and validation code, and understanding which layers should be frozen or fine-tuned. It also helped troubleshoot errors, debug the code, run experiments, and interpret the results to identify possible improvements. Overall, Codex was used as both a learning and coding assistant throughout the project.

## Three prompts, and what we did with the answers

These show how we used the answers: not as final, but as something to question and push further.

> Also, in the research paper, did you specify whether we believe we overfitted or underfitted and explaining why we did or didn't?

Adam asked this after reading a draft that reported accuracy but not whether the models generalise. The answer became section 6.6, which uses our saved training curves to show that the first CNN underfit and that the final model fits its training set without overfitting its decisions.

> No, like, feel free to do a model that takes into consideration the full file byte model and also, like, the visual aspects, a combination. … have a dyno, have conv next, and then third model being the full file byte model, and then Jev either decides on like one hot encodings or maybe on the agreements. Try them all and see what happens and see which one would be better.

The first fusion attempt had not beaten a plain average. Instead of dropping the idea, Adam changed the third expert to one whose mistakes differ from the image models'. That version worked (97.98% against 97.12% on the same images), and the comparison showed that the decision model Jev never beat simple rules.

> is it better to do different thing that just the average of each one or put a rule that makes it like the similar than the minimum amount of taste testers for a poisoned cup problem. also you sure there's nothing we can do or more epochs we can try.

Adam did not accept "averaging is enough" or "we trained long enough" without a test. The poisoned-cup idea became error-correcting output codes, where the minimum number of "tasters" turned out to be the worst rule and redundancy fixed it; the extra epochs became the 4,000-epoch double-descent test of section 6.5.

## What we did ourselves

- **Ideas and direction:** every research question and every idea tested in the paper, as listed above for each member.
- **Decisions:** which experiments to run, which results to trust, the pixel-only rule for the final model, the final model itself, the two entries selected on Kaggle, and what the paper says.
- **Review:** reading and checking the code before using it, re-running experiments, checking every number in the paper against its source, and reviewing each other's work through pull requests.
- **Execution and resources:** downloading the data, our Kaggle accounts and uploads, sharing the daily upload budget, creating and paying for the GPU pod, and running our own notebooks and experiments.
- **Teamwork:** our meetings and discussions, the split of work and the deadlines we kept.
