# FMRec (RecBole-based)

This repo trains **FMRec** using a small RecBole pipeline (without `recbole.quick_start`) via `FMRec/train.py`.

## Requirements

- Python 3.9+ (recommended)
- PyTorch (CPU or CUDA build)
- RecBole

Install dependencies (example):

```bash
pip install -U pip
pip install torch recbole numpy
```

> Note: `FMRec/train.py` patches a few NumPy aliases to keep older RecBole versions working with newer NumPy.

## Dataset

RecBole expects an **atomic** `.inter` file at:

```text
FMRec/datasets/<dataset>/<dataset>.inter
```

For example, if `dataset: steam`, you should have:

```text
FMRec/datasets/steam/steam.inter
```

You can download processed atomic datasets from RecBole’s dataset list page.

## Configure

Edit `FMRec/configs.yaml`:

- `dataset`: dataset folder/file prefix (e.g. `steam`)
- `data_path`: keep as `datasets/` when running inside `FMRec/`
- `load_col.inter`: the column names present in your `.inter` file
- training hyperparams: `epochs`, `train_batch_size`, `learning_rate`, etc.

## Train

Run from the `FMRec/` directory:

```bash
cd FMRec
python train.py
```

Outputs are written under `FMRec/saved/` (best model) and logs under `FMRec/log/` / `FMRec/log_tensorboard/` depending on your config.

