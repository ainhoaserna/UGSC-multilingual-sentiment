"""
inference_pipeline.py
=====================
Zero-shot domain transfer sentiment inference pipeline for the UGSC-ML dataset.

    "Cross-Lingual Sentiment Classification in Sustainable Mobility:
     A Zero-Shot Domain Transfer Evaluation Framework"
    AI (MDPI), 2026 — Serna, Gerrikagoitia, de Oña

Model
-----
CardiffNLP twitter-xlm-roberta-base-sentiment
    https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment

This model is pre-trained on the CC100 corpus and fine-tuned on multilingual
Twitter sentiment data. It is applied here without any task- or domain-specific
fine-tuning (zero-shot domain transfer) to transport-related user reviews.

Algorithm 1 (from paper, Section 2.4)
--------------------------------------
For each input sentence s:
    1. Preprocess s (minimal: punctuation artifacts, Unicode normalisation)
    2. Tokenise using the SentencePiece tokeniser of the model
    3. Run inference → logits for three classes [negative, neutral, positive]
    4. Apply softmax → confidence distribution
    5. Assign predicted class = argmax
    6. Record confidence = max softmax score
    7. If confidence < 0.5 → flag for qualitative taxonomy analysis

Input
-----
ugsc_ml_translations.csv  (wide format, one row per English sentence)
    Required columns: sentence_id, English, Spanish, French, German, Italian

Output
------
sentiment_classification_results.csv  (long format, one row per prediction)
    Columns: sentence_id, Language, Text, English_source, Sentiment, Confidence

Usage
-----
    # Basic usage (runs on CPU):
    python inference_pipeline.py

    # Custom paths:
    python inference_pipeline.py \\
        --input  ugsc_ml_translations.csv \\
        --output sentiment_classification_results.csv \\
        --threshold 0.5 \\
        --batch-size 32

    # GPU if available:
    python inference_pipeline.py --device cuda

Requirements
------------
    pip install transformers torch scipy pandas tqdm
    # CPU-only torch (smaller download):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import argparse
import csv
import os
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

MODEL_NAME = 'cardiffnlp/twitter-xlm-roberta-base-sentiment'

LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Italian']

# Confidence threshold below which a prediction is flagged
# for qualitative taxonomy analysis (Section 2.3 / Algorithm 1, step 7)
DEFAULT_THRESHOLD = 0.5

# Label mapping from model output to paper convention
# The CardiffNLP model returns: 0=Negative, 1=Neutral, 2=Positive
LABEL_MAP = {
    'negative': 'negative',
    'neutral':  'neutral',
    'positive': 'positive',
    # Some model versions use capitalised labels
    'Negative': 'negative',
    'Neutral':  'neutral',
    'Positive': 'positive',
}


# ──────────────────────────────────────────────────────────────
# PREPROCESSING  (Section 2.4)
# ──────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Minimal preprocessing as described in Section 2.4.

    - Unicode NFKC normalisation (accented characters, ligatures)
    - Strip leading/trailing whitespace
    - Remove trailing punctuation artifacts (e.g. '. .' at sentence end)

    No lemmatisation, lowercasing, or stopword removal is applied, to
    preserve the original linguistic structure for XLM-RoBERTa's
    SentencePiece tokeniser.
    """
    # 1. Unicode normalisation
    text = unicodedata.normalize('NFKC', text)
    # 2. Strip whitespace
    text = text.strip()
    # 3. Collapse multiple spaces
    text = ' '.join(text.split())
    return text


# ──────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────

def load_model(device: str = 'cpu'):
    """
    Load the CardiffNLP XLM-RoBERTa sentiment model from Hugging Face.
    The model is downloaded and cached automatically on first run.

    Returns
    -------
    tokeniser, model, softmax function
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        import torch.nn.functional as F
    except ImportError as e:
        raise ImportError(
            'Required packages not found. Install with:\n'
            '    pip install transformers torch'
        ) from e

    print(f'Loading model: {MODEL_NAME}')
    print(f'Device: {device}')
    print('(First run will download ~1 GB from Hugging Face Hub)\n')

    tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    def softmax_fn(logits):
        return F.softmax(logits, dim=-1)

    return tokeniser, model, softmax_fn


# ──────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────

def predict_batch(texts: list[str], tokeniser, model, softmax_fn,
                  device: str = 'cpu', max_length: int = 512) -> list[dict]:
    """
    Run inference on a batch of texts.

    Returns
    -------
    List of dicts with keys: 'sentiment', 'confidence', 'scores'
    """
    import torch

    encoded = tokeniser(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits

    probs = softmax_fn(logits).cpu().numpy()
    id2label = model.config.id2label

    results = []
    for prob_row in probs:
        best_idx = int(prob_row.argmax())
        raw_label = id2label[best_idx]
        sentiment = LABEL_MAP.get(raw_label, raw_label.lower())
        confidence = float(prob_row[best_idx])
        results.append({
            'sentiment':  sentiment,
            'confidence': confidence,
            'scores': {
                LABEL_MAP.get(id2label[i], id2label[i].lower()): float(p)
                for i, p in enumerate(prob_row)
            }
        })
    return results


# ──────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: str,
    output_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 32,
    device: str = 'cpu',
) -> pd.DataFrame:
    """
    Full inference pipeline (Algorithm 1 from paper).

    Parameters
    ----------
    input_path  : Path to wide-format translations CSV.
    output_path : Path to write long-format results CSV.
    threshold   : Confidence threshold for low-confidence flagging.
    batch_size  : Number of sentences per inference batch.
    device      : 'cpu' or 'cuda'.

    Returns
    -------
    DataFrame with all predictions (long format).
    """

    # ── Load input ──
    print(f'Reading input: {input_path}')
    df_wide = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f'  {len(df_wide)} sentences × {len(LANGUAGES)} languages = '
          f'{len(df_wide) * len(LANGUAGES)} predictions\n')

    # ── Load model ──
    tokeniser, model, softmax_fn = load_model(device)

    # ── Build long-format record list ──
    records = []
    for _, row in df_wide.iterrows():
        sid = str(row.get('sentence_id', '')).zfill(4)
        en_source = str(row.get('English', ''))
        for lang in LANGUAGES:
            text = str(row.get(lang, ''))
            if not text.strip():
                continue
            records.append({
                'sentence_id':   sid,
                'Language':      lang,
                'Text':          text,
                'English_source': en_source,
            })

    print(f'Total records to classify: {len(records)}\n')

    # ── Batch inference ──
    all_preds = []
    batches = [records[i:i + batch_size]
               for i in range(0, len(records), batch_size)]

    for batch in tqdm(batches, desc='Classifying', unit='batch'):
        texts = [preprocess(r['Text']) for r in batch]
        preds = predict_batch(texts, tokeniser, model, softmax_fn, device)
        for rec, pred in zip(batch, preds):
            all_preds.append({
                **rec,
                'Sentiment':  pred['sentiment'],
                'Confidence': round(pred['confidence'], 10),
                'low_confidence_flag': pred['confidence'] < threshold,
            })

    # ── Build DataFrame ──
    df_out = pd.DataFrame(all_preds)

    # ── Summary statistics ──
    print('\n── Results summary ──')
    for lang in LANGUAGES:
        lang_df = df_out[df_out['Language'] == lang]
        dist = lang_df['Sentiment'].value_counts()
        lc_n = lang_df['low_confidence_flag'].sum()
        print(
            f'  {lang:<10} '
            f"neg={dist.get('negative', 0):3d}  "
            f"neu={dist.get('neutral',  0):3d}  "
            f"pos={dist.get('positive', 0):3d}  "
            f"| low-conf={lc_n} ({lc_n/len(lang_df)*100:.1f}%)"
        )

    total_lc = df_out['low_confidence_flag'].sum()
    print(f'\n  Total low-confidence (< {threshold}): '
          f'{total_lc} / {len(df_out)} '
          f'({total_lc / len(df_out) * 100:.1f}%)')

    # ── Save ──
    out_cols = ['sentence_id', 'Language', 'Text', 'English_source',
                'Sentiment', 'Confidence', 'low_confidence_flag']
    df_out[out_cols].to_csv(output_path, index=False, encoding='utf-8')
    print(f'\nResults saved to: {output_path}')

    return df_out


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Zero-shot multilingual sentiment inference — UGSC-ML pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input', '-i',
        default='ugsc_ml_translations.csv',
        help='Wide-format input CSV with columns: sentence_id, English, Spanish, '
             'French, German, Italian. (default: ugsc_ml_translations.csv)'
    )
    parser.add_argument(
        '--output', '-o',
        default='sentiment_classification_results.csv',
        help='Output CSV path. (default: sentiment_classification_results.csv)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f'Confidence threshold for low-confidence flagging. '
             f'(default: {DEFAULT_THRESHOLD})'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Inference batch size. Reduce if you run out of memory. '
             '(default: 32)'
    )
    parser.add_argument(
        '--device', '-d',
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Compute device. Use "cuda" for NVIDIA GPU, "mps" for Apple Silicon. '
             '(default: cpu)'
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.input).exists():
        raise FileNotFoundError(
            f'Input file not found: {args.input}\n'
            f'Download the dataset from: https://doi.org/10.5281/zenodo.15085522'
        )

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == '__main__':
    main()
