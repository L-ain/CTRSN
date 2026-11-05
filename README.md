# CTRSN

CTRSN combines trainable class-channel-wise PCA projections with residual shrinkage blocks for EEG-based microsleep classification. This repository contains the data loading, PCA initialization, training, and leave-one-subject-out (LOSO) evaluation workflow for CTRSN.

## Files

| File | Purpose |
| --- | --- |
| `train_ctrsn.py` | Model construction, training, LOSO evaluation, and result export |
| `dataloader.py` | Recording loading, subject grouping, and epoch segmentation |
| `PCA.py` | Per-class, per-channel PCA initialization |
| `ccpca_layer.py` | Trainable projections, reconstruction, and weighted concatenation |
| `AdaptiveFocalLoss.py` | Focal classification loss |

## Environment

Use Python 3.8 with TensorFlow 2.10 and scikit-learn 1.3.2. An example installation is:

```bash
python -m pip install tensorflow==2.10.1 numpy==1.23.5 scikit-learn==1.3.2
```

The implementation uses TensorFlow 2 / Keras 2 APIs. GPU execution requires a compatible CUDA/cuDNN installation. TensorFlow also supports CPU execution. Full training requires the EEG recordings described below.

## Input data

Place preprocessed NumPy recordings in `raw/data/`, or pass another directory through `--data-dir`.

```text
raw/data/
    MC102_recording01.npy
    MC103_recording01.npy
    ...
```

Each file is a numeric array with shape `(12, recording_samples)`:

- Rows 0–10 contain the 11 EEG channels, consistently ordered across recordings.
- Row 11 contains event codes. `0` denotes wakefulness and `2` denotes Type B microsleep. These become classification labels `0` and `1`.
- The first five characters of each filename identify the subject. Files are sorted before grouping.
- The loader includes recordings containing event code `2` and extracts contiguous segments for codes `0` and `2`.

The expected sampling rate is 200 Hz. The input recordings should already have undergone the intended signal preprocessing, including band-pass filtering. This repository loads those recordings and segments them into non-overlapping, label-consistent epochs. Dataset files are supplied separately.

The segmentation implementation excludes the final sample of each contiguous event run and discards trailing samples that cannot form a complete epoch.

## Run training

Run commands from the repository directory:

```bash
python train_ctrsn.py --data-dir raw/data --result-name ctrsn.txt
```

An explicit configuration is:

```bash
python train_ctrsn.py --data-dir raw/data --epoch-samples 100 --pca-components 20 --epochs 10 --batch-size 32 --learning-rate 0.00015 --lambda-rec 1e-8 --seed 0 --result-dir results --result-name ctrsn.txt
```

To enable strict orthogonality of the effective projection matrices:

```bash
python train_ctrsn.py --data-dir raw/data --strict-orthogonal --result-name ctrsn_orthogonal.txt
```

Use `python train_ctrsn.py --help` for the complete command-line interface.

## Workflow

1. Load recordings, group them by subject, and form EEG epochs.
2. Hold out one subject for evaluation and use the remaining subjects for training.
3. Fit separate PCA bases for each class and EEG channel using only the training fold. PCA uses scikit-learn's automatic solver selection. The training seed is passed to PCA.
4. Initialize trainable projection matrices from the PCA components. Each sample is projected into both class-associated subspaces and reconstructed with the corresponding transposed matrix.
5. Compute softmax weights from negative reconstruction errors and concatenate the weighted projected features.
6. Apply three residual shrinkage blocks, flatten the features, and produce two-class softmax probabilities.
7. Optimize the classification and class-conditional reconstruction objectives and evaluate the held-out subject.

For input `(batch, time, channels)`, the projection layer returns features shaped `(batch, channels, 2 * components)`. With default settings this is `(batch, 11, 40)`. The Conv1D layers operate along the channel-index axis of this projected tensor. The PCA fitting means are not applied in the subsequent trainable projection.

Each residual block contains two causal Conv1D layers with kernel size 3. The three blocks use 16, 16, and 8 filters, with dilation rates 1, 2, and 4. The threshold subnetwork uses L1 and L2 kernel regularization with coefficient 0.001. Shortcut projections are applied when feature dimensions differ.

The training objective combines Focal Loss (`alpha=1`, `gamma=1.2`) with two reconstruction terms weighted by `lambda_rec`. Each reconstruction term masks samples by their true class and averages over the entire batch. Classification gradients can reach both projection branches. Early stopping monitors training loss with patience 10 and `restore_best_weights=True`. The default projection matrices are unconstrained. The optional orthogonal variant uses QR decomposition.

## Outputs

With `--result-name ctrsn.txt`, results are written after all eligible folds complete:

```text
results/
    phi/0.5-s/ctrsn.txt
    accuracy/0.5-s/ctrsn.txt
    sensitivity/0.5-s/ctrsn.txt
    specificity/0.5-s/ctrsn.txt
    subjects/0.5-s/ctrsn.txt
```

Metric files contain comma-separated per-subject values. The subject file records the corresponding one-based subject indices. Phi is calculated with scikit-learn's Matthews correlation coefficient. Subjects without microsleep test epochs are skipped. Console averages include all evaluated subjects. The reported inference time is the duration of prediction over a complete held-out subject.

Specify `--result-name` to save results, and use a new filename for every run because output files are created exclusively. The current exporter uses the literal `0.5-s` subdirectory even when `--epoch-samples` is changed. For other epoch lengths, use a separate `--result-dir` and descriptive result filename.
