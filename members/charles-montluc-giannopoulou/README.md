# Charles Montluc-Giannopoulou

Explored transfer learning with a pretrained ResNet18 at a higher input resolution, combined with the file-format information of each PNG.

## Methods tried

| Kaggle entry | Description on Kaggle | Public | Private |
|---|---|---|---|
| `submission.csv` | ImageNet-pretrained ResNet18 fine-tuned on 160px RGB emojis | 0.95889 | 0.96396 |
| `submission.csv` (earlier) | "Correcting output mistake" | 0.78315 | 0.77874 |

## ResNet18 at 160 px with file-format features

Code: [`resnet18-160px-format-features/resnet18_160px_format_features.ipynb`](resnet18-160px-format-features/resnet18_160px_format_features.ipynb), the notebook as it was run on 15 October (outputs kept), and the `submission.csv` it produced. Its data paths point to the Kaggle archive on my laptop; change them to your own copy to rerun it.

- **Model**: ResNet18 pretrained on ImageNet, fully fine-tuned. Each file's original format (colour mode and size, for example `RGBA(240,240)`) is turned into extra features and joined to the CNN output before the last layer.
- **Preprocessing**: transparent pixels composited on a white background, converted to RGB, resized to 160 x 160. The original format is recorded before resizing.
- **Training**: AdamW with a OneCycle learning-rate schedule, 20 epochs, batches of 64, class weights and label smoothing (0.1) in the loss. Augmentation: random flips, small zooms and shifts. Prediction averages each image and its mirror copy.
- **Validation**: 15% of the training set held out, stratified by class.
- **Format correction**: a table of how often each vendor appears for each file format, built on the training split only, multiplies the CNN's probabilities.

| Version | Validation | Kaggle public | Kaggle private |
|---|---|---|---|
| CNN (pixels + format features) | 94.4% | | |
| + format correction (submitted) | 96.4% | 0.95889 | 0.96396 |

## Limits

- The model uses the file-format shortcut described in the paper (data section), so its score is not a pixels-only result and is not comparable with the pixels-only models.
- The format is used twice: as an input of the network and again in the correction step, as if it were separate evidence.
- Training was unstable: validation accuracy fell to 0.71 at epoch 4 and 0.77 at epoch 7 before ending at its best value.
- The submitted model never saw the 15% hold-out, and only one seed was run.

## What I would do next

- A lower learning rate for the pretrained layers (for example 1e-4) than for the last layer.
- Retrain on all 9,879 images, or average models from several folds, and run several seeds.
- Use the format only once (as input or as correction), and test whether flip augmentation hides left/right details such as hand direction.

## AI use

Claude (Anthropic) was used to build the notebook and, on 28 October, to review it and update this repository. See `codebase/AI_USAGE.md`.
