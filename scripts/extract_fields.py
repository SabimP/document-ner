import os
import re
import csv
import json
from typing import List, Tuple, Dict, Optional


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
            items.append(([x1, y1, x3, y3], text))
    return items


# Regexes
DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})[\-/](\d{1,2})[\-/](\d{2,4})\b"),
    re.compile(r"\b(\d{1,2})\s*([A-Za-z]{3,})\s*(\d{2,4})\b"),
    re.compile(r"\b([A-Za-z]{3,})\s*(\d{1,2}),\s*(\d{4})\b"),
]

AMOUNT_PAT = re.compile(r"(?<!\w)(?:RM|USD|SGD|MYR|\$)?\s*\d{1,3}(?:[ ,]\d{3})*(?:\.\d{1,2})?(?!\w)")

def clean_amount(s: str) -> Optional[float]:
    s = s.replace(',', ' ').replace('  ', ' ')
    m = re.search(r"(\d{1,3}(?:[ ]\d{3})*(?:\.\d{1,2})?)", s)
    if not m:
        return None
    try:
        return float(m.group(1).replace(' ', ''))
    except ValueError:
        return None


def extract_date(lines: List[str]) -> Optional[str]:
    for line in lines:
        for pat in DATE_PATTERNS:
            m = pat.search(line)
            if m:
                return m.group(0)
    return None


def extract_invoice_number(lines: List[str]) -> Optional[str]:
    inv_keywords = ["INVOICE", "INV", "INVOICE #", "INVOICE NO", "INVOICE NUMBER"]
    for line in lines:
        u = line.upper()
        if any(k in u for k in inv_keywords):
            m = re.search(r"INVOICE\s*(?:#|NO\.?|NUMBER)?\s*[:\-]?\s*([A-Z0-9\-_/]+)", u)
            if m:
                return m.group(1)
    # fallback: receipt # commonly used
    for line in lines:
        u = line.upper()
        if 'RECEIPT' in u and '#' in u:
            m = re.search(r"RECEIPT\s*#\s*[:\-]?\s*([A-Z0-9\-_/]+)", u)
            if m:
                return m.group(1)
    return None


def extract_amount(lines: List[str]) -> Optional[float]:
    priority_keywords = [
        'TOTAL SALES (INCLUSIVE OF GST)', 'GRAND TOTAL', 'TOTAL AMOUNT',
        'AMOUNT DUE', 'NETT TOTAL', 'TOTAL SALES', 'TOTAL', 'BALANCE DUE'
    ]
    # Priority lines first
    for line in lines:
        u = line.upper()
        if any(k in u for k in priority_keywords):
            # pick the last monetary token on the line
            matches = AMOUNT_PAT.findall(line)
            if matches:
                val = clean_amount(matches[-1])
                if val is not None:
                    return val
    # Fallback: maximum amount seen in document
    cands = []
    for line in lines:
        for m in AMOUNT_PAT.findall(line):
            val = clean_amount(m)
            if val is not None:
                cands.append(val)
    return max(cands) if cands else None


def run(data_root: str, outdir: str):
    labels_dir = os.path.join(data_root, 'labels')
    ids = [os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
    ids.sort()
    rows: List[Dict] = []
    for doc_id in ids:
        path = os.path.join(labels_dir, doc_id + '.txt')
        items = parse_label_file(path)
        lines = [t for _, t in items]
        inv_no = extract_invoice_number(lines)
        date = extract_date(lines)
        amount = extract_amount(lines)
        rows.append({
            'doc_id': doc_id,
            'invoice_number': inv_no,
            'date': date,
            'amount': amount,
        })
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, 'field_extraction.csv')
    json_path = os.path.join(outdir, 'field_extraction.json')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['doc_id', 'invoice_number', 'date', 'amount'])
        w.writeheader()
        w.writerows(rows)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print(f'Wrote: {csv_path}')
    print(f'Wrote: {json_path}')


if __name__ == '__main__':
    # default paths relative to repo
    here = os.path.dirname(os.path.dirname(__file__))
    data_root = os.path.join(here, 'dataset')
    outdir = os.path.join(here, 'outputs')
    run(data_root, outdir)
