import argparse
import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import shutil
try:
    from torchcrf import CRF
except ImportError:
    # Some environments expose the package as 'TorchCRF'
    from TorchCRF import CRF

from transformers import (
    LayoutLMForTokenClassification,
    LayoutLMTokenizerFast,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# =============================
# Data parsing for provided dataset (images + simple label txt with quad coords + text)
# Each label line appears to be: x1,y1,x2,y2,x3,y3,x4,y4,text
# We don't have entity tags, so we define a synthetic NER-like tagging by weak rules:
# - Numeric-like tokens -> NUMBER
# - Contains ':' or '#' or looks like field label -> FIELD
# - Uppercase alphabetic sequences -> ORG/HEADER
# - Else -> OTHER
# This is for educational comparison only. If you have gold NER tags, plug them in.
# =============================

LABEL_LIST = ["O", "B-FIELD", "I-FIELD", "B-NUM", "I-NUM", "B-HEADER", "I-HEADER"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

IMG_SIZE = (1000, 1000)  # normalize boxes into this canvas (LayoutLM requires 0..1000)


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class BoxText:
    text: str
    box: List[int]  # [x0,y0,x1,y1]
    label: str


def parse_label_file(path: str) -> List[Tuple[List[int], str]]:
    items: List[Tuple[List[int], str]] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 9:
                continue
            try:
                x1, y1, x2, y2, x3, y3, x4, y4 = map(int, parts[:8])
                text = ','.join(parts[8:]).strip()
            except Exception:
                continue
            # convert quad to bbox
            xs = [x1, x2, x3, x4]
            ys = [y1, y2, y3, y4]
            x0, y0, x1b, y1b = min(xs), min(ys), max(xs), max(ys)
            items.append(([x0, y0, x1b, y1b], text))
    return items


def rule_label_token(tok: str) -> str:
    t = tok.strip()
    if not t:
        return 'O'
    # numeric
    if any(c.isdigit() for c in t):
        return 'B-NUM'
    # field-like keys
    triggers = [':', '#', 'INVOICE', 'TOTAL', 'AMOUNT', 'DATE', 'TIME', 'TAX', 'QTY', 'PRICE', 'CHANGE', 'CASH']
    if any(k in t.upper() for k in triggers):
        return 'B-FIELD'
    # header-like uppercase tokens
    if t.isupper() and len(t) >= 3:
        return 'B-HEADER'
    return 'O'


def normalize_box(box: List[int], size: Tuple[int, int]) -> List[int]:
    x0, y0, x1, y1 = box
    W, H = size
    # clamp
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = max(x0, x1), max(y0, y1)
    # scale to 0..1000
    return [
        int(x0 / W * 1000),
        int(y0 / H * 1000),
        int(x1 / W * 1000),
        int(y1 / H * 1000),
    ]


class FUNSDLikeDatasetLayoutLM(Dataset):
    def __init__(self, root: str, split_ids: List[str], tokenizer: LayoutLMTokenizerFast, max_len: int = 512):
        self.root = root
        self.ids = split_ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.root, 'images', img_id + '.jpg')
        label_path = os.path.join(self.root, 'labels', img_id + '.txt')

        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        items = parse_label_file(label_path)

        words = []
        boxes = []
        labels = []
        for box, text in items:
            label_for_box = rule_label_token(text)
            # tokenize text by whitespace; assign same bbox per token
            for j, tok in enumerate(text.split()):
                words.append(tok)
                boxes.append(normalize_box(box, (W, H)))
                # BIO for multi-token fields/numbers/headers (simplified)
                base = label_for_box
                if base.startswith('B-'):
                    tag = 'B-' + base[2:]
                else:
                    tag = base
                labels.append(LABEL2ID.get(tag, 0))

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_offsets_mapping=False,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
        )
        # align boxes/labels to word_ids
        word_ids = encoding.word_ids()
        aligned_boxes = []
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_boxes.append([0, 0, 0, 0])
                aligned_labels.append(-100)
            else:
                aligned_boxes.append(boxes[word_id])
                lab = labels[word_id]
                # set I- for subsequent wordpieces of same word
                if word_id == prev_word_id and lab != -100 and lab != 0:
                    # convert B-XXX to I-XXX id
                    b_label = ID2LABEL[lab]
                    if b_label.startswith('B-'):
                        i_label = 'I-' + b_label[2:]
                        lab = LABEL2ID.get(i_label, lab)
                aligned_labels.append(lab)
            prev_word_id = word_id

        encoding['bbox'] = aligned_boxes
        encoding['labels'] = aligned_labels
        return {k: torch.tensor(v) for k, v in encoding.items()}


class FUNSDLikeDatasetTextOnly(Dataset):
    def __init__(self, root: str, split_ids: List[str], tokenizer: AutoTokenizer, max_len: int = 512):
        self.root = root
        self.ids = split_ids
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        label_path = os.path.join(self.root, 'labels', img_id + '.txt')
        # concatenate all line texts to a doc sequence
        items = parse_label_file(label_path)
        words = []
        labels = []
        for box, text in items:
            box_label = rule_label_token(text)
            toks = text.split()
            for j, tok in enumerate(toks):
                words.append(tok)
                tag = box_label
                if j > 0 and tag.startswith('B-'):
                    tag = 'I-' + tag[2:]
                labels.append(LABEL2ID.get(tag, 0))
        encoding = self.tok(
            words,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_offsets_mapping=False,
        )
        word_ids = encoding.word_ids()
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            else:
                lab = labels[word_id]
                if word_id == prev_word_id and lab != -100 and lab != 0:
                    b_label = ID2LABEL[lab]
                    if b_label.startswith('B-'):
                        i_label = 'I-' + b_label[2:]
                        lab = LABEL2ID.get(i_label, lab)
                aligned_labels.append(lab)
            prev_word_id = word_id
        encoding['labels'] = aligned_labels
        return {k: torch.tensor(v) for k, v in encoding.items()}


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_labels: int, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        # TorchCRF API: CRF(num_labels, pad_idx=None, use_gpu=True)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, crf_mask=None):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        emissions = self.classifier(x)
        # Build mask: prefer provided crf_mask; otherwise fall back to attention_mask or non-pad
        if crf_mask is not None:
            mask = crf_mask.bool()
        elif attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = input_ids.ne(self.pad_idx)
        if labels is not None:
            # TorchCRF returns log-likelihood per batch element; take mean negative for loss
            llh = self.crf(emissions, labels, mask=mask)
            loss = -llh.mean()
            return loss
        else:
            pred = self.crf.viterbi_decode(emissions, mask=mask)
            return pred


def make_splits(all_ids: List[str], seed=42) -> Tuple[List[str], List[str]]:
    ids = sorted(all_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(0.8 * n)
    return ids[:n_train], ids[n_train:]


def make_splits_3(all_ids: List[str], seed: int = 42, val_ratio: float = 0.1, test_ratio: float = 0.2) -> Tuple[List[str], List[str], List[str]]:
    """Create train/val/test splits. Defaults: 70/10/20.
    """
    ids = sorted(all_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_test = int(test_ratio * n)
    n_val = int(val_ratio * (n - n_test))
    test_ids = ids[:n_test]
    remain = ids[n_test:]
    val_ids = remain[:n_val]
    train_ids = remain[n_val:]
    return train_ids, val_ids, test_ids


def collect_ids(root: str) -> List[str]:
    imgs = [f for f in os.listdir(os.path.join(root, 'images')) if f.lower().endswith('.jpg')]
    ids = [os.path.splitext(f)[0] for f in imgs]
    return ids


def _compute_accuracy_from_preds_labels(all_preds: List[int], all_labels: List[int]) -> float:
    if not all_labels:
        return 0.0
    correct = sum(int(p == y) for p, y in zip(all_preds, all_labels))
    return 100.0 * correct / len(all_labels)


def _eval_layoutlm(model, loader: DataLoader, device: str) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch['labels'].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            preds = logits.argmax(-1)
            mask = batch['attention_mask'].detach().cpu().numpy()
            for i in range(preds.shape[0]):
                for j in range(preds.shape[1]):
                    if mask[i, j] == 1 and labels[i, j] != -100:
                        all_preds.append(int(preds[i, j]))
                        all_labels.append(int(labels[i, j]))
    return all_preds, all_labels


def _eval_bilstm(model, loader: DataLoader, device: str) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            labels = batch['labels']
            crf_mask = (attn.cpu() == 1) & (labels != -100)
            pred_paths = model(input_ids, attention_mask=attn, labels=None, crf_mask=crf_mask.to(device))
            for i, path in enumerate(pred_paths):
                mask = attn[i].detach().cpu().numpy()
                labs = labels[i].numpy()
                for j, p in enumerate(path):
                    if j < len(mask) and mask[j] == 1 and labs[j] != -100:
                        all_preds.append(int(p))
                        all_labels.append(int(labs[j]))
    return all_preds, all_labels


def _plot_accuracy_curves(curves: Dict[str, List[float]], title: str, out_path: str):
    plt.figure(figsize=(7, 4))
    epochs = range(1, len(next(iter(curves.values()))) + 1) if curves else []
    for k, v in curves.items():
        plt.plot(epochs, v, label=k)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, out_path: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, f'{val}', horizontalalignment="center",
                     color="white" if val > thresh else "black", fontsize=7)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_layoutlm(data_root: str, outdir: str, epochs: int = 5, batch_size: int = 2, lr: float = 5e-5, max_len: int = 512, device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = LayoutLMTokenizerFast.from_pretrained('microsoft/layoutlm-base-uncased')
    model = LayoutLMForTokenClassification.from_pretrained(
        'microsoft/layoutlm-base-uncased',
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    ids = collect_ids(data_root)
    train_ids, val_ids, test_ids = make_splits_3(ids)

    train_ds = FUNSDLikeDatasetLayoutLM(data_root, train_ids, tokenizer, max_len)
    val_ds = FUNSDLikeDatasetLayoutLM(data_root, val_ids, tokenizer, max_len)
    test_ds = FUNSDLikeDatasetLayoutLM(data_root, test_ids, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * num_training_steps), num_training_steps)

    logs = {"train_loss": [], "train_acc": [], "val_acc": [], "test_acc": []}
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f'LayoutLM Train {ep+1}/{epochs}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_loader))
        logs["train_loss"].append(float(avg_loss))

        # Epoch accuracies
        tr_preds, tr_labels = _eval_layoutlm(model, train_loader, device)
        va_preds, va_labels = _eval_layoutlm(model, val_loader, device)
        te_preds, te_labels = _eval_layoutlm(model, test_loader, device)
        tr_acc = _compute_accuracy_from_preds_labels(tr_preds, tr_labels)
        va_acc = _compute_accuracy_from_preds_labels(va_preds, va_labels)
        te_acc = _compute_accuracy_from_preds_labels(te_preds, te_labels)
        logs["train_acc"].append(tr_acc)
        logs["val_acc"].append(va_acc)
        logs["test_acc"].append(te_acc)
        print(f'Epoch {ep+1} loss: {avg_loss:.4f} | acc T/V/Te: {tr_acc:.1f}/{va_acc:.1f}/{te_acc:.1f}%')

    os.makedirs(outdir, exist_ok=True)
    model.save_pretrained(os.path.join(outdir, 'layoutlm-model'))
    tokenizer.save_pretrained(os.path.join(outdir, 'layoutlm-model'))
    with open(os.path.join(outdir, 'layoutlm_logs.json'), 'w') as f:
        json.dump(logs, f, indent=2)
    _plot_accuracy_curves(
        {"Train": logs["train_acc"], "Validation": logs["val_acc"], "Test": logs["test_acc"]},
        'LayoutLM accuracy per epoch', os.path.join(outdir, 'layoutlm_accuracy.png')
    )

    # Eval
    all_preds, all_labels = _eval_layoutlm(model, test_loader, device)

    rep = classification_report(all_labels, all_preds, target_names=LABEL_LIST, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    # Plot confusion for embedding in report
    _plot_confusion_matrix(cm, LABEL_LIST, 'LayoutLM Confusion', os.path.join(outdir, 'layoutlm_confusion.png'))
    return rep, cm, (train_ids, val_ids, test_ids), logs


def build_tokenizer_vocab(data_root: str, min_freq: int = 1, pretrained_name: str = 'bert-base-uncased'):
    # Use BERT tokenizer but restrict to word-level tokens; CRF model consumes input_ids from tokenizer vocab
    tok = AutoTokenizer.from_pretrained(pretrained_name)
    return tok


def train_bilstm_crf(data_root: str, outdir: str, epochs: int = 5, batch_size: int = 8, lr: float = 1e-3, max_len: int = 256, device: str = None, splits: Tuple[List[str], List[str], List[str]] = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = build_tokenizer_vocab(data_root)
    pad_idx = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    ids = collect_ids(data_root)
    if splits is not None:
        train_ids, val_ids, test_ids = splits
    else:
        train_ids, val_ids, test_ids = make_splits_3(ids)

    train_ds = FUNSDLikeDatasetTextOnly(data_root, train_ids, tokenizer, max_len)
    val_ds = FUNSDLikeDatasetTextOnly(data_root, val_ids, tokenizer, max_len)
    test_ds = FUNSDLikeDatasetTextOnly(data_root, test_ids, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = BiLSTMCRF(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, num_labels=len(LABEL_LIST), pad_idx=pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logs = {"train_loss": [], "train_acc": [], "val_acc": [], "test_acc": []}
    model.train()
    for ep in range(epochs):
        total = 0.0
        for batch in tqdm(train_loader, desc=f'BiLSTM+CRF Train {ep+1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            labels = batch['labels'].clone()
            # Build CRF mask: valid where labels != -100 and attention is 1
            crf_mask = (attn.cpu() == 1) & (labels != -100)
            # Replace -100 with a valid label id (e.g., 0) to satisfy CRF target constraints
            labels[labels == -100] = 0
            labels = labels.to(device)
            loss = model(input_ids, attention_mask=attn, labels=labels, crf_mask=crf_mask.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
        avg_loss = total / max(1, len(train_loader))
        logs["train_loss"].append(float(avg_loss))

        # Epoch accuracies
        tr_preds, tr_labels = _eval_bilstm(model, train_loader, device)
        va_preds, va_labels = _eval_bilstm(model, val_loader, device)
        te_preds, te_labels = _eval_bilstm(model, test_loader, device)
        tr_acc = _compute_accuracy_from_preds_labels(tr_preds, tr_labels)
        va_acc = _compute_accuracy_from_preds_labels(va_preds, va_labels)
        te_acc = _compute_accuracy_from_preds_labels(te_preds, te_labels)
        logs["train_acc"].append(tr_acc)
        logs["val_acc"].append(va_acc)
        logs["test_acc"].append(te_acc)
        print(f'Epoch {ep+1} loss: {avg_loss:.4f} | acc T/V/Te: {tr_acc:.1f}/{va_acc:.1f}/{te_acc:.1f}%')

    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(outdir, 'bilstm_crf.pt'))
    with open(os.path.join(outdir, 'bilstm_crf_logs.json'), 'w') as f:
        json.dump(logs, f, indent=2)
    _plot_accuracy_curves(
        {"Train": logs["train_acc"], "Validation": logs["val_acc"], "Test": logs["test_acc"]},
        'BiLSTM+CRF accuracy per epoch', os.path.join(outdir, 'bilstm_accuracy.png')
    )

    # Eval
    all_preds, all_labels = _eval_bilstm(model, test_loader, device)

    rep = classification_report(all_labels, all_preds, target_names=LABEL_LIST, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, LABEL_LIST, 'BiLSTM+CRF Confusion', os.path.join(outdir, 'bilstm_confusion.png'))
    return rep, cm, logs


def tabulate_results(name: str, rep: Dict) -> Dict:
    # Extract overall macro avg and per-class F1
    overall = rep.get('macro avg', {})
    out = {
        'model': name,
        'precision': overall.get('precision', 0.0),
        'recall': overall.get('recall', 0.0),
        'f1': overall.get('f1-score', 0.0),
        'support': overall.get('support', 0),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join('dataset'))
    parser.add_argument('--outdir', type=str, default='outputs')
    parser.add_argument('--layoutlm_epochs', type=int, default=3)
    parser.add_argument('--bilstm_epochs', type=int, default=5)
    parser.add_argument('--layoutlm_batch', type=int, default=2)
    parser.add_argument('--bilstm_batch', type=int, default=8)
    parser.add_argument('--max_len_layoutlm', type=int, default=512)
    parser.add_argument('--max_len_bilstm', type=int, default=256)
    args = parser.parse_args()

    seed_all(42)

    print('Training LayoutLM...')
    rep_l, cm_l, (train_ids, val_ids, test_ids), logs_l = train_layoutlm(
        args.data_root, args.outdir, epochs=args.layoutlm_epochs, batch_size=args.layoutlm_batch, max_len=args.max_len_layoutlm
    )

    print('Training BiLSTM+CRF...')
    rep_b, cm_b, logs_b = train_bilstm_crf(
        args.data_root, args.outdir, epochs=args.bilstm_epochs, batch_size=args.bilstm_batch, max_len=args.max_len_bilstm, splits=(train_ids, val_ids, test_ids)
    )

    # Save detailed reports
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, 'layoutlm_report.json'), 'w') as f:
        json.dump(rep_l, f, indent=2)
    with open(os.path.join(args.outdir, 'bilstm_crf_report.json'), 'w') as f:
        json.dump(rep_b, f, indent=2)

    # Confusion matrices
    np.save(os.path.join(args.outdir, 'layoutlm_confusion.npy'), cm_l)
    np.save(os.path.join(args.outdir, 'bilstm_crf_confusion.npy'), cm_b)

    # Comparison table
    row_l = tabulate_results('LayoutLM', rep_l)
    row_b = tabulate_results('BiLSTM+CRF', rep_b)
    df = pd.DataFrame([row_l, row_b])
    df.to_csv(os.path.join(args.outdir, 'comparison.csv'), index=False)
    print('\nComparison summary:')
    print(df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    # Also print split for reproducibility
    with open(os.path.join(args.outdir, 'split.json'), 'w') as f:
        json.dump({'train': train_ids, 'val': val_ids, 'test': test_ids}, f, indent=2)

    # Generate a structured report with required sections
    def _copy_sample_images(src_dir: str, dst_dir: str, k: int = 3) -> List[str]:
        os.makedirs(dst_dir, exist_ok=True)
        imgs = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        imgs.sort()
        chosen = imgs[:k]
        out_paths = []
        for name in chosen:
            sp = os.path.join(src_dir, name)
            dp = os.path.join(dst_dir, name)
            try:
                shutil.copyfile(sp, dp)
                out_paths.append(os.path.relpath(dp, args.outdir).replace('\\', '/'))
            except Exception:
                pass
        return out_paths

    # Derive dataset stats (counts)
    def _dataset_stats(data_root: str) -> Dict:
        image_dir = os.path.join(data_root, 'images')
        label_dir = os.path.join(data_root, 'labels')
        img_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
        # Token-level class counts using simple parse + rule labels
        cls_counts = {l: 0 for l in LABEL_LIST}
        total_tokens = 0
        for f in img_files:
            lab = os.path.join(label_dir, os.path.splitext(f)[0] + '.txt')
            if not os.path.isfile(lab):
                continue
            for box, text in parse_label_file(lab):
                label_for_box = rule_label_token(text)
                toks = text.split()
                for j, _ in enumerate(toks):
                    tag = label_for_box
                    if j > 0 and tag.startswith('B-'):
                        tag = 'I-' + tag[2:]
                    cls_counts[tag] = cls_counts.get(tag, 0) + 1
                    total_tokens += 1
        return {
            'num_images': len(img_files),
            'class_counts': cls_counts,
            'total_tokens': total_tokens,
        }

    stats = _dataset_stats(args.data_root)
    samples = _copy_sample_images(os.path.join(args.data_root, 'images'), os.path.join(args.outdir, 'samples'), k=3)

    # Extract numeric accuracies
    test_acc_layoutlm = float(rep_l.get('accuracy', 0.0) * 100.0 if isinstance(rep_l.get('accuracy', 0.0), float) else rep_l.get('accuracy', 0.0))
    test_acc_bilstm = float(rep_b.get('accuracy', 0.0) * 100.0 if isinstance(rep_b.get('accuracy', 0.0), float) else rep_b.get('accuracy', 0.0))

    def _pct(x: float) -> str:
        try:
            return f"{x:.2f}%"
        except Exception:
            return str(x)

    report_md = f"""# FUNSD-like Document NER Training Report"""

    with open(os.path.join(args.outdir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write(report_md)


if __name__ == '__main__':
    main()
