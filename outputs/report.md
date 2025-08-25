
# Report: Document NER on FUNSD-like Invoices

## Introduction (5 Marks)

### Objective of the coursework (Research questions) (2)
- Can layout-aware transformers (LayoutLM) outperform a text-only BiLSTM+CRF baseline for token-level field extraction on invoice-like documents?
- How do preprocessing choices (bounding-box normalization, weak labels) affect downstream accuracy and errors?

### Real-World Problem and impact (2)
Information extraction from business documents (invoices, receipts, forms) reduces manual data entry, improves back-office automation, and increases accuracy in finance and procurement. Image processing and deep learning convert pixel-level scans into structured data (dates, totals, invoice numbers), enabling faster workflows and fewer errors.

### Overview of the report (1)
We describe the dataset and preprocessing, present two models (LayoutLM and BiLSTM+CRF), outline the training/validation/testing procedure, and report results with accuracy curves and confusion matrices, followed by critical analysis and conclusions.

## Creative and innovative approaches (10 Marks)

### Innovative and original approaches (4)
- Compare a layout-aware model (LayoutLM) that jointly consumes text and bounding boxes against a lightweight BiLSTM+CRF text-only baseline.
- Use weak heuristics to synthesize labels (FIELD/NUM/HEADER/O) in absence of gold NER tags, enabling experimentation without manual annotation.
- Produce end-to-end, reproducible pipeline with automated plots, per-epoch curves, and a structured report.

### Methods or strategies proposed (4)
- Architectures: LayoutLMForTokenClassification vs. BiLSTM with CRF decoding.
- Preprocessing: parse quad coordinates, normalize to 0..1000 for LayoutLM; whitespace tokenization; BIO expansion across wordpieces.
- Optimization: AdamW for LayoutLM with linear warmup; Adam for BiLSTM+CRF; gradient clipping; fixed seeds.

### Justification for suitability (2)
Invoices are spatially structured; LayoutLM exploits layout cues, generally outperforming text-only models. BiLSTM+CRF provides a simple, strong baseline with transparent decoding.

## Simulations (25 Marks)

### Dataset description and samples (5)
- Images: 360 total; synthetic token classes: O, B-FIELD, I-FIELD, B-NUM, I-NUM, B-HEADER, I-HEADER.
- Token count: 41085
- Class distribution (tokens): {'O': 773, 'B-FIELD': 3731, 'I-FIELD': 3812, 'B-NUM': 11184, 'I-NUM': 12091, 'B-HEADER': 3655, 'I-HEADER': 5839}

Sample images:

![](samples/X00016469670.jpg)
![](samples/X00016469671.jpg)
![](samples/X51005200931.jpg)

### Encoding and preprocessing (5)
- Convert quadrilateral label coordinates to axis-aligned boxes and normalize to 0..1000 for LayoutLM.
- Tokenize by whitespace; align subwords with BIO tags; ignore special tokens with label -100.
- Text-only pipeline drops boxes but preserves token labels for CRF.

### Network architecture, training/validation/testing, learning algorithm (15)
- Splits: Train 260 | Val 28 | Test 72 (70/10/20 approx.).
- LayoutLM: transformer encoder with token classification head; AdamW + linear warmup; batch size 2; epochs 3.
- BiLSTM+CRF: 2x directions LSTM hidden 256, CRF decoding; Adam; batch size 8; epochs 5.
- Accuracies recorded each epoch on train/val/test; losses logged for training diagnostics.

## Results Obtained (15 Marks)

1) Test set accuracy (5)
- LayoutLM test accuracy: 97.81%
- BiLSTM+CRF test accuracy: 86.94%

2) Accuracy curves (5)

LayoutLM per-epoch accuracy:
![](layoutlm_accuracy.png)

BiLSTM+CRF per-epoch accuracy:
![](bilstm_accuracy.png)

3) Confusion matrices with explanation (5)

LayoutLM confusion:
![](layoutlm_confusion.png)

BiLSTM+CRF confusion:
![](bilstm_confusion.png)

Interpretation: Diagonal dominance indicates correct predictions; off-diagonal mass between B-FIELD/I-FIELD and O suggests boundary/labeling ambiguity introduced by weak labels; Layout cues reduce confusion for headers and numeric fields.

## Critical Analysis of Results (10 Marks)

### How results were achieved and avenues to improve (5)
- Achieved via layout-aware modeling, careful alignment of tokens/boxes, and CRF decoding. Improvements: replace weak labels with gold annotations; tune hyperparameters; larger epochs; use LayoutLMv3 or LayoutXLM; augment OCR/noise; curriculum learning.

### Factors affecting performance and optimization potential (5)
- Label noise from heuristics caps performance; tokenization granularity impacts BIO alignment; limited data size increases variance. Potential: better label propagation, self-training, multi-task learning (key-value linking), adding visual features (Donut, DocFormer), and post-processing rules for dates/amounts.

## Conclusions (5 Marks)

### Restatement and summary (3)
We studied invoice token classification with two approaches. LayoutLM, leveraging spatial information, outperformed a text-only BiLSTM+CRF baseline on synthetic labels.

### Key takeaways (2)
- Layout matters for documents; even simple baselines benefit from structured preprocessing.
- High-quality labels dominate outcomes; invest in annotation or robust weak supervision.
- The pipeline provides plots, metrics, and artifacts for rapid iteration.
