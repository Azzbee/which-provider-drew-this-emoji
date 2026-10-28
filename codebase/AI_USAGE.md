# AI usage

Team The Winners: Adam Zeraiki, Lina Oumhand, Charles Montluc-Giannopoulou, Erdem Sanver. IDSI51006 Computer Vision, emoji vendor classification.

I keep this record for the team. The course allows AI for code and for understanding the models, provided every use is declared and the prompts are kept with the codebase, and our teacher also allowed it in class for writing the paper. This file says which tools each of us used, what they did, the prompts that shaped the work, and what we did ourselves.

## How we used AI

The questions and the ideas came from us: what to try, why, and what would count as an answer. We used AI the way we would use a fast, tireless engineer and a sparring partner: to turn an idea into working code, to run it, to point out where an idea could be made sharper, and to help us write it up. Every number in the paper was checked against the files that produced it, and we are responsible for everything in it.

A few examples of how an idea of ours was refined, not replaced:

- **One model per idea, then combine them.** I wanted several different models with a transformer in the middle deciding between them, as in TRIBE v2. AI built it; it did not beat a plain average, and that is in the paper. I then asked to swap the third model for one that reads only the PNG file bytes, so the experts would make different kinds of mistakes. That version worked (97.98% against 97.12% on the same images).
- **The poisoned-cup puzzle.** I came across it while preparing for interviews and asked whether its logic could combine our experts. AI identified the machine-learning version of the idea (error-correcting output codes) and tested it: the minimum number of "tasters" is the worst rule, and redundancy makes it work.
- **Double descent.** I read Nakkiran et al. out of curiosity and wanted to know whether it applies to our task. AI replicated their protocol on our data, up to 4,000 epochs and 11 network sizes.
- **Fitting.** I asked whether we overfitted or underfitted and why; the answer, built from our saved training curves, is a section of the paper.

## Tools

| Tool | Who | For what |
|---|---|---|
| Claude (Anthropic), in claude.ai and Claude Code | Adam, Charles, Erdem | code, experiments, GPU runs, repository, paper drafting |
| OpenAI ChatGPT and Codex | Adam, Lina | baseline notebook, DINOv2, fusion and decision-model experiments, Lina's two methods |
| Jev (TypeSafe), Julia-1 | Adam | tested as decision models inside the experiments, not as assistants |

## What each of us did with AI

### Adam (me)

| Date | Tool | What it did |
|---|---|---|
| 30 Sept | Claude | Drafted my first-submission Keras notebook (`members/adam-zeraiki/first-submission-keras-cnn/`). |
| 30 Sept | OpenAI Codex | Generated and debugged the baseline PyTorch CNN notebook (`members/adam-zeraiki/baseline-cnn/`). |
| 17 Oct | Claude | Wrote the codebase (`src/emoji_provider`, scripts, notebook, tests) and ran the data audit, the metadata diagnostic and the EmojiCNNv2 experiments. |
| 17 to 18 Oct | Claude, in my browser | Ran the two ConvNeXt-Tiny GPU runs on the RunPod pod I created and paid for, built the checkpoint average and uploaded the run B file to Kaggle when I asked. |
| 17 to 18 Oct | OpenAI Codex | Implemented and ran my DINOv2 fine-tuning idea (the DINOv2 entry was submitted by Lina), the style features, the TRIBE v2-style fusion transformer and MLP, and the Julia-1 and Jev decision models (`members/adam-zeraiki/`, `members/lina-oumhand/dinov2-partial-finetune/`). |
| 18 Oct | Claude | Drafted the first report from the codebase's results. |
| 27 to 28 Oct | Claude | Wrote the LaTeX paper and its figures from our saved outputs, set up the GitHub repository, and built the file-byte model (`members/adam-zeraiki/file-forensics-shortcut-model/`). |
| 28 Oct | Claude, in my browser and on my Mac | Ran my three-expert fusion (DINOv2 + ConvNeXt + file-byte model), the combination rules including the poisoned-cup codes, and prepared the Jev experiment, which I ran from my own computer with my own key (`members/adam-zeraiki/three-expert-fusion-dinov2-convnext-filebytes/`). Ran the double-descent replication on my RunPod account, $2.42 in total (`members/adam-zeraiki/double-descent-2000-epochs/`). Reviewed and merged my teammates' pull requests with me. |

### Erdem

| Date | Tool | What it did |
|---|---|---|
| 13 to 28 Oct | Claude Code | Experiment series E1a to E16 (`members/erdem-sanver/experiment-series-E1-E16/`). Erdem set the goals and constraints, chose to keep the model pixel-only rather than use the file-format shortcut, approved the plan, reviewed the results at each stage and decided which files to submit. Claude implemented, debugged and ran the code (preprocessing, baselines, CNN and ConvNeXt training, ensembles, pseudo-labelling), drafted the write-ups and, on 28 Oct, added the records to the repository and the paper. |

### Charles

| Date | Tool | What it did |
|---|---|---|
| ~15 Oct | Claude | Built the ResNet18 notebook (160 px, file-format features, format correction). Prompts not kept. |
| 28 Oct | Claude, in VS Code | Reviewed his notebook, wrote `members/charles-montluc-giannopoulou/README.md` and corrected the description of his model in the paper. |
| 28 Oct | Claude | Checked the notebook for keys and data, then added it unchanged with its `submission.csv` to `members/charles-montluc-giannopoulou/resnet18-160px-format-features/`. |

### Lina

| Date | Tool | What it did |
|---|---|---|
| ~17 Oct | OpenAI Codex | Wrote and ran the Extra Trees baseline, the hand-crafted style features, the frozen ResNet18 feature extraction and the SVM selection (`members/lina-oumhand/extra-trees-and-resnet18-svm/`); Lina reviewed the results and submitted the files. |
| 28 Oct | OpenAI Codex | Added her files, the ablation and her original prompts to the repository. |
| 28 Oct | OpenAI Codex | Removed the duplicate method-folder AI-use file at Lina's request and redirected documentation links to this shared record. |

## Prompts

Quoted as written, apart from punctuation. I kept the prompts that shaped the work and left out short confirmations ("go", "ok do it") and messages about files or permissions. Where a prompt included the course brief or other pasted material, it is shown as "…".

### Adam

**30 September, first submission (Claude and Codex)**

> look at this and make a suggestion for a first submissions off of an ipynb that I could run on colab or vs code locally

**17 October, codebase and experiments (Claude)**

> can you look at the folder there is information in it pertaining to the assignment for computer vision as well as all the course information. Here is the explanation of the work, please help me do something that's absolutely phenomenal : … and here is how the paper should be written : Sections for the project for comp vision: 1. Intro/Front page 2. Summary (abstract), table of contents + model chosen (is it a CNN, is it a VLM, etc…) 3. Preprocessing (data augmentation?) 4. Architecture of the model 5. (optional : Metrics Chosen) 6. Performance 7. AI and Code 8. (optional : Conclusion)

> It's open now. Have a look. But before you launch one, please just check the price and give me an estimate of the price before, okay? Thanks.

> don't forget to cut the runpod once you finished and extracted all the relevant information including the loss graphs, validation, etc... to put into the paper at the end

**17 to 18 October, DINOv2, fusion and decision models (Codex)**

> Okay cool. Can you try fine-tuning a meta model? For example, the Dino V3, and fine-tuning it on this, and making the entire process, and then giving me just the CSV I can upload. It would be really cool things.

> Would it be interesting to introduce a similar architecture to TriV2, where you have three input models, one transformer in the middle, and then that gives, that's your source of truth that makes, like, the final decision? So we have, like, for example, like Dino V2, ConvNext, and, like, another model, and then in the middle there's a transformer that takes the output of the first three models, and then from that makes its final decision. Would that be more interesting, and would that potentially give a better answer?

> ok perfect, then do it. Also test it with a jev like model and we'll see if it gives interesting results

> ok let's try it and then also try the jev as the transformer at the end that makrs the decision

**18 October, final entries and first report (Claude)**

> Okay, cool. Can you please write the entire report and I will review?

**27 to 28 October, paper and repository (Claude)**

> Can you now write it as a proper research paper using latex? Also if you want you can check the github that was used for the preprint for the Inria teaching spectra paper. Add also pictures to help explain and graphs and stuff. Make it phenomenal

> Did you also explain the double descent as well as the different approaches we tried and stuff. Also explaining the data preprocessing as well as the Dino v3 etc…

> Okay, cool. Could you also talk about the... Should we put all the different approaches we tried in the appendix also? Or would it be better to include them throughout the paper? Like, for example, like, we tried this, didn't work, so we modified the idea to this. And also explain the try model thing we tried with the Jev, the Dino, and the extra model with the middle transformer, essentially trying to translate the things. And also include like pictures or like clear demonstrations of the architecture of the models, you know? And also like in the beginning explain like different types of like EMLs, for example, like convolutional neural networks, transformers, etc. Showing like just to make it as comprehensive as possible and explaining it as best as possible.

> Also, could you then afterwards push it to create a new GitHub repo, push it to the GitHub repo, and then give me the link with like short message I could send to the group, and make sure the GitHub repo is public, just because I want my other teammates to be able to work on it and push the thing to GitHub so they can modify the research paper. So like the breakdown of the thing, also something explaining for the agents, like making the agents.md just for the other participants to be able to, the other members of the team be able to understand. Also, could you add in as well, like folders inside of it, one folder per person with like the different method each person tried. It would be pretty cool, thanks.

> Also, in the research paper, did you specify whether we believe we overfitted or underfitted and explaining why we did or didn't?

> I want you to create a model with Jev that uses the 3.2 what the files reveal before any pixel is read and I want it to get at least 99% just to show if we wanted to we could've just done that if we just wanted to get a high score but decided not to. … Finally review the github and everything and get an AI_Usage.md ready and make sure everything is ready for a final submission. Also, see if there are any improvements you want to make, show it to me and I will let you know whether to add them in or not.

**28 October, three experts, combination rules and double descent (Claude)**

> No, like, feel free to do a model that takes into consideration the full file byte model and also, like, the visual aspects, a combination. Kind of like, you know my multimodal idea where I take several different models and different things and potentially have, like, Jev do the final decision, potentially do the same thing with that, you know? And instead have also a mod, like, have a dyno, have conv next, and then third model being the full file byte model, and then Jev either decides on like one hot encodings or maybe on the agreements. Try them all and see what happens and see which one would be better.

> Also, feel free to do like a lot more epochs. Like, for example, in Double Descent, like with the thing that they did, they went like, there's 2,000 epochs and things like that, you know? So if you want to reach more, you can go for it. And if you want to use the run pod or whatever, you can go for it.

> is it better to do different thing that just the average of each one or put a rule that makes it like the similar than the minimum amount of taste testers for a poisoned cup problem. also you sure there's nothing we can do or more epochs we can try. Also this is the double descent paper : …

> sure then would it be nice to say that I came across the poisoned cup idea from my revision for interviews and double descent from the paper I read out of curiosity and cite the paper to see if it would apply in this case?

### Erdem

Erdem's requests, summarised by him (not the verbatim prompts):

> Build the experiment series from scratch within the competition constraints: improve the model one change at a time, validate locally before spending a shared submission, log every run in the course spreadsheet so the report can show what each change did, and keep the model pixel-only. Then push the score as far as it goes, write up every experiment in detail, and add the work to the team repository through a pull request.

The experiment plan was set in discussion between Erdem and Claude Code, starting from Erdem's constraints, and approved by Erdem before any code was written. It covered a metadata-only diagnostic, colour histogram + HOG with an SVM, a small CNN from scratch with and without alpha compositing, ImageNet ResNet18 frozen and fully fine-tuned, EfficientNet-B0 and ConvNeXt-Tiny, an augmentation study, higher input resolution, soft-vote ensembles and retraining on all data. In a later stage, which Erdem delegated, a controlled bicubic versus nearest-neighbour resize test, a three-model ensemble and pseudo-labelling of confident test images were added.

### Charles

Charles's request, summarised by him (not the verbatim prompt):

> Please include the modifications inside the report, modify the AI Usage to include this Prompt, and add my modifications in the readme

## What we did ourselves

**The ideas and the direction.**
- I proposed fine-tuning a self-supervised model (DINO) for the task, the TRIBE v2-style fusion of several models with a transformer in the middle, trying decision models such as Jev as the final arbiter, and, when those did not beat a plain average, replacing the third expert with a model that never looks at the pixels.
- I brought the poisoned-cup puzzle from my interview preparation and the double-descent paper from my own reading, and asked for both to be tested on our data; I also pushed for training far longer (2,000 and then 4,000 epochs) to see whether a second descent would appear.
- I asked whether we were overfitting or underfitting, and asked to check how far the file-format shortcut alone could go.
- I asked for the paper to tell the story as "we tried this, it did not work, so we changed it", for the model families to be explained from the ground up, and for a public repository with one folder per member so everyone could contribute.
- Erdem set the constraints of his series (one change at a time, validate locally before using a shared submission, log every run, stay pixel-only), approved its plan and chose what to submit.
- Charles chose to explore transfer learning at a higher input resolution.
- Lina chose to start from a simple Extra Trees baseline, then set the goal of improving on its 0.81838 score.

**The decisions.**
- Which experiments to run, which results to trust and which to drop, and which model to submit as final.
- Keeping the final model pixel-only, even though the file-format shortcut was there to use.
- Which two entries to select for the private leaderboard, and how to share the five daily submissions between the four of us.
- What goes into the paper and what stays in the appendix, and accepting or rejecting each pull request.

**The resources and the running.**
- Downloading the competition data, our Kaggle accounts and every upload except the run B file, which Claude uploaded from my browser when I asked.
- Creating, paying for and stopping the RunPod GPU pod, with the price checked before every launch.
- Sending the 615 Jev requests from my own computer with my own API key.

**The checking.**
- Reviewing every number in the paper against the saved outputs, reading the drafts and asking for corrections.
- Reviewing each other's work through pull requests before merging.

**The teamwork.**
- Our meetings and discussions, the split of work between the four of us, and the deadlines we kept.
