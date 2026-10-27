# Lina Oumhand

Explored the self-supervised Vision Transformer route (DINOv2).

## Methods tried

| Kaggle entry | Method | Public | Private |
|---|---|---|---|
| `submission_dinov2_finetuned.csv` | DINOv2 fine-tuned | 0.95586 | 0.95607 |
| `submission2.csv` | not recorded | 0.96719 | 0.96983 |
| `submission.csv` | not recorded | 0.81838 | 0.83259 |

The paper discusses DINOv2 in Section 4.4 and why it trailed ConvNeXt here (pretraining invariances, patch geometry), marked as untested hypotheses.

## To complete

- DINOv2 settings: model size (ViT-S/14, B/14, ...), input resolution, epochs, learning rate, which layers were trained, validation score and split. Confirm whether it was DINOv2 or DINOv3.
- What `submission.csv` and `submission2.csv` were (submission2 is your best entry).
- Add your notebook here, and send Adam the AI tools and prompts you used (he keeps the AI-use record).
