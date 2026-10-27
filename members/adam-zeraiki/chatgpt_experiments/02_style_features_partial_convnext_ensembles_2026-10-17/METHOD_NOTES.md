# Experiment notes — 17 October 2026

These are AI-assisted technical experiment records, not a student-authored final report.

## Evaluation design

All supervised model comparisons use the same supplied split: 7,903 training images and 1,976 held-out images. The earlier DINOv2 checkpoint was already selected on this holdout, and this extension compares several additional candidates. Scores are therefore exploratory model-selection results. They are not measured Kaggle scores, and uncertainty intervals after selection do not eliminate selection bias.

The seven labels identify image providers: apple, facebook, google, messenger, mozilla, samsung and whatsapp. No test labels, filename-derived labels, source-file dimensions, file-size predictors or train/test matching shortcuts are used. Deterministic white-background alpha compositing and aspect-preserving resizing retain the complete emoji. Exact RGB96 and actual RGB224 duplicate groups are checked against the outer split; frozen-model inner cross-validation unions both equality relations.

## Preserved experiments

- The original DINOv2 model was reconstructed from official pretrained weights plus its actual fine-tuned final two blocks and head. Original validation probabilities were reproduced bit-for-bit.
- One fixed test-time augmentation averaged original and horizontally flipped image probabilities. The flip was applied to the rendered input, with actual model inference.
- Two simple classifiers used fixed pixel, colour, edge, alpha and texture features. Neither uses external provider metadata. Their fixed DINO blends are all retained, including inferior attempts.
- ConvNeXt-Tiny uses official TorchVision ImageNet weights. Its early stages are frozen; its final stage, output normalization and seven-provider classifier are genuinely updated by backpropagation. The prefix cache is deterministic; the trainable stage retains the model's stochastic-depth regularizer. No image augmentation is applied in this run.
- A frozen ConvNeXt feature baseline selects its linear classifier regularization only inside grouped folds of the training partition.
- Fixed DINOv2/ConvNeXt probability averages test complementary errors. No combiner is fitted on validation labels, and no large blending-weight sweep is performed.

## Reporting conventions

The original DINOv2 score is 1,895/1,976 = 95.9008%. One additional correct validation image equals about 0.0506 percentage points. The practical model-selection threshold for added complexity is at least eight additional correct images over that reference. This threshold is not a statistical significance test.

Final comparisons use the same explicit log-loss clipping convention (1e-12), because scikit-learn's float32 default clipping can yield a different number for identical probabilities. Paired error corrections and regressions are more informative than comparing independently rounded accuracies.

All selected checkpoints retain the measured train/validation split. There is no all-label refit in this batch. No Kaggle submission was performed; existing daily submission limits were left untouched.
