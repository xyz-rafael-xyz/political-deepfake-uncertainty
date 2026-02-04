# Political Deepfake Uncertainty (reproducible repo)

This repository is a **full, files-separated project** that reproduces the full experiment pipeline:
- stream **OpenFake** from Hugging Face
- deterministic **metadata keyword filtering** to create a political subset
- write images + `metadata.csv`
- create fixed `train/val/test` splits (2800/600/600)
- train **ResNet-18** and **EfficientNet-B4** end-to-end
- run deterministic inference, temperature scaling, MC dropout (T=1 + T>1)
- compute metrics (Acc/AUC/ECE/Brier/NLL + confusion counts)
- generate the plots used in Results (ROC, reliability, score hist, confusion matrix, uncertainty diagnostics)

---

## 0) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 1) Build the political subset (OpenFake → filtered dataset)

```bash
python scripts/data/01_download_filter.py \
  --keywords_yaml configs/political_keywords.yaml \
  --out_dir data/political_openfake \
  --resize 380,380
```

This creates:
- `data/political_openfake/images/*.png`
- `data/political_openfake/metadata.csv`

## 2) Make splits (2800/600/600)

```bash
python scripts/data/02_make_splits.py \
  --metadata_csv data/political_openfake/metadata.csv \
  --out_dir data/political_openfake/splits \
  --train_n 2800 --val_n 600 --test_n 600 --seed 42
```

## 3) Train models (end-to-end)

EfficientNet-B4:
```bash
python scripts/train/01_train.py --backbone efficientnet_b4 --seed 42 --resolution 380
```

ResNet-18:
```bash
python scripts/train/01_train.py --backbone resnet18 --seed 42 --resolution 380
```

Checkpoints saved under `runs/<backbone>_seed<seed>_res<res>/train/best.pt`.

## 4) Produce predictions (val + test)

```bash
# EfficientNet
python scripts/eval/01_predict_det.py --backbone efficientnet_b4 \
  --ckpt runs/efficientnet_b4_seed42_res380/train/best.pt \
  --split_csv data/political_openfake/splits/val.csv \
  --out_dir runs

python scripts/eval/01_predict_det.py --backbone efficientnet_b4 \
  --ckpt runs/efficientnet_b4_seed42_res380/train/best.pt \
  --split_csv data/political_openfake/splits/test.csv \
  --out_dir runs
```

Repeat for `resnet18`.

## 5) Temperature scaling

```bash
python scripts/eval/03_temperature_scaling.py \
  --val_pred_dir runs/efficientnet_b4/predictions \
  --test_pred_dir runs/efficientnet_b4/predictions
```

This writes `p_temp.npy` into the test prediction folder.

## 6) MC Dropout (T=20 + T=1)

```bash
python scripts/eval/04_mc_dropout.py \
  --backbone efficientnet_b4 \
  --ckpt runs/efficientnet_b4_seed42_res380/train/best.pt \
  --split_csv data/political_openfake/splits/test.csv \
  --T 20
```

Creates:
- `p_mc_mean_T20.npy`
- `var_mc_T20.npy`
- `ent_mc_T20.npy`
- and `p_mc_T1.npy`

## 7) Results figures (each subsection has its own script)

Example (confusion matrix):
```bash
python scripts/results/5_1_confusion_matrices.py \
  --pred_dir runs/efficientnet_b4/predictions \
  --out_dir figures/eff/ --prefix fig1_
```

ROC:
```bash
python scripts/results/5_2_roc.py \
  --pred_dir runs/efficientnet_b4/predictions \
  --out_dir figures/eff/ --prefix fig2_ --method det
```

Reliability:
```bash
python scripts/results/5_5_reliability.py \
  --pred_dir runs/efficientnet_b4/predictions \
  --out_dir figures/eff/ --prefix fig3_ --method det
```

Entropy separation + uncertainty-error ROC:
```bash
python scripts/results/5_8_entropy_sep.py --pred_dir runs/efficientnet_b4/predictions --out_dir figures/eff/ --prefix fig6_ --T 20
python scripts/results/5_9_uncertainty_error_roc.py --pred_dir runs/efficientnet_b4/predictions --out_dir figures/eff/ --prefix fig7_ --T 20
```

---

## Repo layout

- `src/polideep/` core library
- `scripts/data/` dataset build & split
- `scripts/train/` training
- `scripts/eval/` prediction + MC + temperature scaling
- `scripts/results/` **one script per Results subsection** figure/table

