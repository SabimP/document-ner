# FUNSD-like NER comparison: LayoutLM vs BiLSTM+CRF

This project trains and evaluates two token classification models on your dataset in `dataset/`:
- LayoutLM (text + bounding boxes)
- BiLSTM+CRF (text only)

Label files are parsed from `dataset/labels/*.txt` where each line appears as:
```
x1,y1,x2,y2,x3,y3,x4,y4,text
```
Since gold entity tags are not present, the code applies simple weak rules to derive synthetic labels (FIELD/NUM/HEADER/O). Replace `rule_label_token()` with your gold labels if available.

## Setup (Windows PowerShell)

```
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

## Train and evaluate

```
python train_funsd.py --data_root dataset --outdir outputs --layoutlm_epochs 3 --bilstm_epochs 5
```

Artifacts are written to `outputs/`:
- layoutlm_model/ and bilstm_crf.pt
- layoutlm_report.json, bilstm_crf_report.json
- layoutlm_confusion.npy, bilstm_crf_confusion.npy
- comparison.csv (precision/recall/F1 macro)
- split.json (train/test IDs)

## Notes
- LayoutLM expects bounding boxes normalized to 0..1000; the code converts from image pixel coordinates.
- BiLSTM+CRF uses a BERT tokenizer's vocab for convenience.
- Replace the labeling heuristic with true NER tags to get meaningful metrics.
