# First-submission CNN (30 September)

Keras CNN at 64 px: three Conv2D/MaxPool stages (32, 64, 128 channels), dropout 0.4, dense 128, softmax over 7 vendors. Adam, batch 64, up to 15 epochs, 15% stratified hold-out, early stopping on validation accuracy (patience 3). Optional second input with native size and PNG mode (`USE_SIZE_AND_MODE`).

Result (from the notebook's notes; the saved notebook has no executed outputs): about 81% validation with pixels only, about 91% with size and mode.

File: `first_submission_30sep.ipynb`.
