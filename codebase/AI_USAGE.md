# AI usage record

Team The Winners (Adam Zeraiki, Lina Oumhand, Charles Montluc-Giannopoulou, Erdem Sanver), IDSI51006 Computer Vision.

The course allows AI for the code and for understanding the models, provided its use is disclosed and the prompts are kept in the codebase. The teacher also allowed AI for writing the paper, in class. This file lists every AI tool used on the project, what it did and the prompts given to it. Prompts are quoted exactly; messages that only granted permissions, pointed to file locations or handled repository housekeeping are left out.

## 1. Summary

| Date | Tool | Member | What the AI did |
|---|---|---|---|
| 30 Sept | Claude (Anthropic) | Adam | Drafted the first-submission Keras notebook (`members/adam-zeraiki/first-submission-keras-cnn/`). |
| 30 Sept | OpenAI Codex | Adam | Generated and debugged the baseline PyTorch notebook (`members/adam-zeraiki/baseline-cnn/`). |
| 13 to 14 Oct | not recorded | Erdem | Experiments E1a to E15. **Erdem to confirm** whether an AI tool was used. |
| ~15 Oct | Claude (Anthropic) | Charles | Built the ResNet18 notebook (160 px, file-format features, format correction). Prompts not kept. |
| 28 Oct | Claude (Anthropic) | Charles | Modified the GitHub repository: added notes about Charles's own work (`members/charles-montluc-giannopoulou/README.md`) and corrected the description of his model in the paper (who-tried-what row, journey table step 6, Kaggle table) after a review of his notebook. |
| ~17 Oct | not recorded | Lina | Extra-trees and frozen ResNet18 + SVM entries. **Lina to confirm** whether an AI tool was used. |
| 17 Oct | Claude (Anthropic) | Adam | Wrote this codebase (`src/emoji_provider`, scripts, notebook, tests); ran the EDA, the metadata diagnostic, the baseline re-score and the EmojiCNNv2 experiments; produced the figures and number tables. |
| 17 to 18 Oct | Claude (Anthropic), driving Adam's browser | Adam | Ran both ConvNeXt-Tiny GPU runs on a RunPod pod that Adam created and paid for; added the last-epoch option; built the 10-checkpoint average; submitted the run B file on Kaggle and selected the entries, at Adam's request. |
| 17 to 18 Oct | OpenAI Codex | Adam | Wrote and ran the DINOv2 partial fine-tuning, style features, partially tuned ConvNeXt, learned fusion (transformer and MLP), Julia-1 and Jev decision-model experiments (`members/adam-zeraiki/` and `members/lina-oumhand/dinov2-partial-finetune/`). |
| 18 Oct | Claude (Anthropic) | Adam | Drafted the first report from the codebase's results. |
| 27 to 28 Oct | Claude (Anthropic) | Adam | Wrote the LaTeX paper and its figures from the saved outputs; verified Kaggle scores on the competition pages; recovered the experiment records; built the GitHub repository; built the file-forensics model (`members/adam-zeraiki/file-forensics-shortcut-model/`); wrote this file. |
| 28 Oct | Claude (Anthropic), driving Adam's browser | Adam | Ran the three-expert fusion (DINOv2 + ConvNeXt + file-byte model) and prepared the three-expert Jev experiment (`members/adam-zeraiki/three-expert-fusion-dinov2-convnext-filebytes/`); signed in to Adam's RunPod account through his Google session at his request, restarted his pod, ran the 2,000- and 4,000-epoch double-descent experiments (two batches, $2.42) and stopped the pod; tested combination rules beyond averaging (`combine_rules.py`) (`members/adam-zeraiki/double-descent-2000-epochs/`); opened the Jev launcher on Adam's Mac (Adam entered his own API key). |

The team chose the experiments and the final model, reviewed every number against the saved outputs and is responsible for the content.

## 2. Prompts

### 30 September, Claude: first-submission notebook

> look at this and make a suggestion for a first submissions off of an ipynb that I could run on colab or vs code locally

(with the project instructions e-mail and the example notebook attached)

### 30 September, OpenAI Codex: baseline notebook

> look at this and make a suggestion for a first submissions off of an ipynb that I could run on colab or vs code locally :

(followed by the teacher's project e-mail)

> ok cool take control of my chrome and make it happen here is the data

### 17 October, Claude: codebase and experiments

> can you look at the folder there is information in it pertaining to the assignment for computer vision as well as all the course information. Here is the explanation of the work, please help me do something that's absolutely phenomenal : [the teacher's project e-mail] and here is how the paper should be written : Sections for the project for comp vision: 1. Intro/Front page 2. Summary (abstract), table of contents + model chosen (is it a CNN, is it a VLM, etc…) 3. Preprocessing (data augmentation?) 4. Architecture of the model 5. (optional : Metrics Chosen) 6. Performance 7. AI and Code 8. (optional : Conclusion)

(with a photo of the whiteboard listing the same sections)

> I have access to the databricks metalab gpu

> which submission.csv file should I upload?

> [screenshot of the Kaggle submissions page] here are the scores. feel free to take control of my chrome to check and use databricks

> Okay, can you cancel that one and just use run pod instead?

> It's open now. Have a look. But before you launch one, please just check the price and give me an estimate of the price before, okay? Thanks.

> go

> don't forget to cut the runpod once you finished and extracted all the relevant information including the loss graphs, validation, etc... to put into the paper at the end

### 17 to 18 October, OpenAI Codex: DINOv2, fusion and decision models

> Okay cool. Can you try fine-tuning a meta model? For example, the Dino V3, and fine-tuning it on this, and making the entire process, and then giving me just the CSV I can upload. It would be really cool things.

> Okay, cool. I can't upload it unfortunately because I only have five uploads a day and we've already done five. I think the team has also uploaded until we hit the limit. However, feel free to carry on and do a bunch of other things as well and see if potentially we'll have a better score or better model or etc., etc.

> Would it be interesting to introduce a similar architecture to TriV2, where you have three input models, one transformer in the middle, and then that gives—that's your source of truth that makes, like, the final decision? So we have, like, for example, like Dino V2, ConvNext, and, like, another model, and then in the middle there's a transformer that takes the output of the first three models, and then from that makes its final decision. Would that be more interesting, and would that potentially give a better answer?

> I meant tribe v2

> ok perfect, then do it. Also test it with a jev like model and we'll see if it gives interesting results

> no i meant jev as in the new decision models

> ok let's try it and then also try the jev as the transformer at the end that makrs the decision

### 18 October, Claude: final Kaggle entries and first report

> also can you do it all quickly we need to submit in an hour

> let's do one last one come on

> ok do it

> ok let's prep the research paper to write

> write it

> Okay, cool. Can you please write the entire report and I will review?

### 27 to 28 October, Claude: LaTeX paper and repository

> Can you now write it as a proper research paper using latex? Also if you want you can check the github that was used for the preprint for the Inria teaching spectra paper. Add also pictures to help explain and graphs and stuff. Make it phenomenal

> Did you also explain the double descent as well as the different approaches we tried and stuff. Also explaining the data preprocessing as well as the Dino v3 etc…

> Okay, cool. Could you also talk about the... Should we put all the different approaches we tried in the appendix also? Or would it be better to include them throughout the paper? Like, for example, like, we tried this, didn't work, so we modified the idea to this. And also explain the try model thing we tried with the Jev, the Dino, and the extra model with the middle transformer, essentially trying to translate the things. And also include like pictures or like clear demonstrations of the architecture of the models, you know? And also like in the beginning explain like different types of like EMLs, for example, like convolutional neural networks, transformers, etc. Showing like just to make it as comprehensive as possible and explaining it as best as possible.

> Also, could you then afterwards push it to create a new GitHub repo, push it to the GitHub repo, and then give me the link with like short message I could send to the group, and make sure the GitHub repo is public, just because I want my other teammates to be able to work on it and push the thing to GitHub so they can modify the research paper. So like the breakdown of the thing, also something explaining for the agents, like making the agents.md just for the other participants to be able to, the other members of the team be able to understand. Also, could you add in as well, like folders inside of it, one folder per person with like the different method each person tried. It would be pretty cool, thanks.

> Also, in the research paper, did you specify whether we believe we overfitted or underfitted and explaining why we did or didn't?

> Here is all the work I did with ChatGPT: [link to the Google Drive folder of the experiment records]

> Also what's the max amount of epochs we did

> I want you to create a model with Jev that uses the 3.2 what the files reveal before any pixel is read and I want it to get at least 99% just to show if we wanted to we could've just done that if we just wanted to get a high score but decided not to. Also, lina sent me this, modify her section please. [Lina's description of her two submissions] Finally review the github and everything and get an AI_Usage.md ready and make sure everything is ready for a final submission. Also, see if there are any improvements you want to make, show it to me and I will let you know whether to add them in or not.

### 28 October, Claude: three-expert fusion and 2,000-epoch run

> No, like, feel free to do a model that takes into consideration the full file byte model and also, like, the visual aspects, a combination. Kind of like, you know my multimodal idea where I take several different models and different things and potentially have, like, Jev do the final decision, potentially do the same thing with that, you know? And instead have also a mod, like, have a dyno, have conv next, and then third model being the full file byte model, and then Jev either decides on like one hot encodings or maybe on the agreements. Try them all and see what happens and see which one would be better.

> Also, feel free to do like a lot more epochs. Like, for example, in Double Descent, like with the thing that they did, they went like, there's 2,000 epochs and things like that, you know? So if you want to reach more, you can go for it. And if you want to use the run pod or whatever, you can go for it.

> Okay, do all that yourself.

> is it better to do different thing that just the average of each one or put a rule that makes it like the similar than the minimum amount of taste testers for a poisoned cup problem. also you sure there's nothing we can do or more epochs we can try. Also this is the double descent paper : [title and abstract of Nakkiran et al., Deep Double Descent]

> Also, can you check Charles's PR and see whether to accept it or not? And did you add the information for Lina that she sent me or not?

(Adam then pasted his TypeSafe API key in the chat. Claude did not use it: it prepared a launcher script and opened it on Adam's Mac at his request, "take control of my screen we'll do it", and Adam pasted the key into his own Terminal, which sent the 615 Jev requests.)

> Yes it's okay for the logs

(approving the download of the training logs from the RunPod pod to his Mac)

### 28 October, Claude: Charles's notes and paper corrections

Charles's request, summarised by him afterwards (not the verbatim prompt):

> Please include the modifications inside the report, modify the AI Usage to include this Prompt, and add my modifications in the readme

The content came from a review of his notebook, run with Claude in VS Code.

## 3. What was not done by AI

Downloading the data from Kaggle, the Kaggle accounts and uploads (except the run B upload, made by Claude at Adam's request from his browser), creating, paying for and stopping the RunPod pod, choosing to keep the final model pixel-only, choosing which experiments to run, and the team's discussions are the team's own work.
