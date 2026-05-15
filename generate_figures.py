"""
generate_figures.py
===================
Generates all three figures from the paper:

    "Cross-Lingual Sentiment Classification in Sustainable Mobility:
     A Zero-Shot Domain Transfer Evaluation Framework"
    AI (MDPI), 2026 — Serna, Gerrikagoitia, de Oña

Figures generated
-----------------
- Figure 1: Donut chart — overall sentiment class distribution (n = 1,875)
- Figure 2: Stacked bar chart — sentiment distribution per language (n = 375 per language)
- Figure 3: Combined bar + dot chart — low-confidence case distribution and mean confidence
             by linguistic pattern (n = 113 categorized cases)

Input files required
--------------------
- sentiment_classification_results_1875.csv
    Columns: Language, Sentiment, Confidence, (others)
    Rows: 1,875 (375 per language × 5 languages)

- low_confidence_annotated_113.csv
    Columns: final_category, Confidence
    Rows: 113 categorized low-confidence cases (confidence < 0.5)
    Note: excludes Other/Unclear category

Usage
-----
    python generate_figures.py

    # Custom input paths:
    python generate_figures.py \\
        --results path/to/results.csv \\
        --lowconf path/to/lowconf.csv \\
        --outdir  path/to/output/

Output
------
All figures are saved to ./figures/ (created if it does not exist).
    figures/figure1_donut.png
    figures/figure2_stacked_bar.png
    figures/figure3_combined.png

Requirements
------------
    pip install matplotlib pandas numpy
"""

import os
import argparse
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ──────────────────────────────────────────────────────────────
# COLOUR PALETTE  (consistent across all three figures)
# ──────────────────────────────────────────────────────────────
C_NEG = '#C0392B'   # red   — Negative
C_NEU = '#7F8C8D'   # grey  — Neutral
C_POS = '#2980B9'   # blue  — Positive

# Category colours for Figure 3 (ordered by frequency, ascending)
CAT_COLORS = {
    'Irony / Sarcasm':            '#8E44AD',
    'Informal Punctuation':       '#566573',
    'Conditional / Hypothetical': '#1A5276',
    'Idiomatic Expressions':      '#117A65',
    'Mixed Sentiment':            '#B7950B',
    'Translation Drift':          '#922B21',
}

# Display order for Figure 3 (ascending frequency so bars grow left→right)
CATEGORY_ORDER = [
    'Irony / Sarcasm',
    'Informal Punctuation',
    'Conditional / Hypothetical',
    'Idiomatic Expressions',
    'Mixed Sentiment',
    'Translation Drift',
]

# Line-broken labels for Figure 3 y-axis
CATEGORY_LABELS = [
    'Irony /\nSarcasm',
    'Informal\nPunctuation',
    'Conditional /\nHypothetical',
    'Idiomatic\nExpressions',
    'Mixed\nSentiment',
    'Translation\nDrift',
]

LANGUAGES = ['English', 'French', 'German', 'Italian', 'Spanish']


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────

def load_results(path: str) -> list[dict]:
    """Load sentiment classification results CSV."""
    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    print(f"  Loaded {len(rows)} rows from {path}")
    return rows


def load_lowconf(path: str) -> list[dict]:
    """Load annotated low-confidence cases CSV."""
    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    print(f"  Loaded {len(rows)} rows from {path}")
    return rows


# ──────────────────────────────────────────────────────────────
# FIGURE 1 — Donut chart: overall sentiment distribution
# ──────────────────────────────────────────────────────────────

def plot_figure1(rows: list[dict], outpath: str) -> None:
    """
    Figure 1: Donut chart showing overall sentiment class distribution
    across all 1,875 predictions (all languages combined).
    """
    overall = Counter(r['Sentiment'] for r in rows)
    neg = overall['negative']
    neu = overall['neutral']
    pos = overall['positive']
    grand = neg + neu + pos

    sizes  = [neg, neu, pos]
    colors = [C_NEG, C_NEU, C_POS]
    labels = ['Negative', 'Neutral', 'Positive']
    pcts   = [n / grand * 100 for n in sizes]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    fig.patch.set_facecolor('white')

    ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.52, edgecolor='white', linewidth=2.5),
        counterclock=False,
    )

    # Centre annotation
    ax.text(0,  0.07, f'n = {grand:,}', ha='center', va='center',
            fontsize=11, color='#555555', fontfamily='DejaVu Sans')
    ax.text(0, -0.18, 'predictions',   ha='center', va='center',
            fontsize=9,  color='#888888', fontfamily='DejaVu Sans')

    # Legend with counts and percentages
    legend_labels = [
        f'{lbl}  {n:,}  ({p:.1f}%)'
        for lbl, n, p in zip(labels, sizes, pcts)
    ]
    patches = [mpatches.Patch(facecolor=c, edgecolor='white') for c in colors]
    ax.legend(patches, legend_labels,
              loc='lower center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=10,
              handlelength=1.2, handleheight=1.0)

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {outpath}")


# ──────────────────────────────────────────────────────────────
# FIGURE 2 — Stacked horizontal bar: distribution per language
# ──────────────────────────────────────────────────────────────

def plot_figure2(rows: list[dict], outpath: str) -> None:
    """
    Figure 2: Stacked horizontal bar chart showing the proportional
    distribution of predicted sentiment classes per language
    (n = 375 per language).
    """
    # Build per-language counts
    neg_counts = []
    neu_counts = []
    pos_counts = []
    for lang in LANGUAGES:
        lr = [r for r in rows if r['Language'] == lang]
        d  = Counter(r['Sentiment'] for r in lr)
        neg_counts.append(d['negative'])
        neu_counts.append(d['neutral'])
        pos_counts.append(d['positive'])

    n_per_lang = 375
    neg_pct = [n / n_per_lang * 100 for n in neg_counts]
    neu_pct = [n / n_per_lang * 100 for n in neu_counts]
    pos_pct = [n / n_per_lang * 100 for n in pos_counts]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor('white')

    ax.barh(LANGUAGES, neg_pct, color=C_NEG, label='Negative', height=0.55)
    ax.barh(LANGUAGES, neu_pct, left=neg_pct, color=C_NEU, label='Neutral', height=0.55)
    ax.barh(LANGUAGES, pos_pct,
            left=[neg_pct[i] + neu_pct[i] for i in range(len(LANGUAGES))],
            color=C_POS, label='Positive', height=0.55)

    # Value labels inside (or outside for very small segments)
    for i in range(len(LANGUAGES)):
        # Negative
        ax.text(neg_pct[i] / 2, i,
                f'{neg_counts[i]}\n({neg_pct[i]:.0f}%)',
                ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')

        # Neutral — label outside if segment < 8 %
        if neu_pct[i] >= 8:
            ax.text(neg_pct[i] + neu_pct[i] / 2, i,
                    f'{neu_counts[i]}\n({neu_pct[i]:.0f}%)',
                    ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        else:
            ax.text(neg_pct[i] + neu_pct[i] + 1.2, i,
                    f'{neu_counts[i]} ({neu_pct[i]:.0f}%)',
                    ha='left', va='center',
                    fontsize=7.5, color='#444444')

        # Positive
        ax.text(neg_pct[i] + neu_pct[i] + pos_pct[i] / 2, i,
                f'{pos_counts[i]}\n({pos_pct[i]:.0f}%)',
                ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')

    ax.set_xlabel('Percentage of predictions (%)', fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax.set_axisbelow(True)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0, labelsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.legend(loc='lower right', frameon=True, fontsize=9,
               framealpha=0.9, edgecolor='#cccccc', ncol=3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {outpath}")


# ──────────────────────────────────────────────────────────────
# FIGURE 3 — Combined bar + dot: low-confidence taxonomy
# ──────────────────────────────────────────────────────────────

def plot_figure3(lc_rows: list[dict], outpath: str) -> None:
    """
    Figure 3: Two-panel figure.
    Left panel  — horizontal bar chart: percentage of categorised
                  low-confidence cases per linguistic pattern.
    Right panel — dot plot: mean confidence score per pattern.
    Patterns are sorted by frequency (ascending).
    n = 113 categorised cases (Other/Unclear excluded).
    """
    n_total = len(lc_rows)

    # Compute N, %, mean confidence per category
    n_cases  = []
    pct_vals = []
    conf_vals = []
    colors   = []

    for cat in CATEGORY_ORDER:
        cat_rows = [r for r in lc_rows if r['final_category'] == cat]
        n = len(cat_rows)
        pct = n / n_total * 100
        safe_confs = []
        for r in cat_rows:
            try:
                safe_confs.append(float(r['Confidence']))
            except (ValueError, TypeError):
                pass
        mean_conf = sum(safe_confs) / len(safe_confs) if safe_confs else 0.0

        n_cases.append(n)
        pct_vals.append(pct)
        conf_vals.append(mean_conf)
        colors.append(CAT_COLORS[cat])

    # Weighted mean confidence (for reference line)
    weighted_mean = sum(c * n for c, n in zip(conf_vals, n_cases)) / sum(n_cases)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4.2),
        gridspec_kw={'width_ratios': [2, 1]}
    )
    fig.patch.set_facecolor('white')

    # ── Left panel: bar chart ──
    ax1.barh(CATEGORY_LABELS, pct_vals, color=colors, height=0.55)
    for i, (p, n) in enumerate(zip(pct_vals, n_cases)):
        ax1.text(p + 0.3, i, f'{p:.1f}%  (n={n})',
                 va='center', fontsize=8.5, color='#333333')

    ax1.set_xlabel('% of categorized low-confidence cases', fontsize=9)
    ax1.set_xlim(0, 43)
    ax1.xaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax1.set_axisbelow(True)
    for sp in ['top', 'right', 'left']:
        ax1.spines[sp].set_visible(False)
    ax1.tick_params(axis='y', length=0, labelsize=9)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.set_title('Distribution of categorized cases',
                  fontsize=10, pad=8, fontweight='bold')

    # ── Right panel: dot plot ──
    ax2.scatter(conf_vals, range(len(CATEGORY_ORDER)), color=colors, s=130, zorder=5)
    x_min = min(conf_vals) - 0.006
    for i, c in enumerate(conf_vals):
        ax2.plot([x_min, c], [i, i], color='#cccccc', linewidth=1.5, zorder=1)
        ax2.text(c + 0.0008, i, f'{c:.3f}',
                 va='center', fontsize=8.5, color='#333333')

    ax2.axvline(x=weighted_mean, color='#888888',
                linestyle=':', linewidth=1.2, alpha=0.8,
                label=f'Mean: {weighted_mean:.3f}')
    ax2.set_xlabel('Mean confidence score', fontsize=9)
    ax2.set_xlim(x_min, max(conf_vals) + 0.012)
    ax2.set_yticks([])
    ax2.xaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    ax2.set_axisbelow(True)
    for sp in ['top', 'right', 'left']:
        ax2.spines[sp].set_visible(False)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.set_title('Mean confidence per pattern',
                  fontsize=10, pad=8, fontweight='bold')
    ax2.legend(fontsize=8, frameon=True, framealpha=0.9,
               edgecolor='#cccccc', loc='lower right')

    plt.tight_layout(w_pad=2)
    plt.savefig(outpath, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {outpath}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate figures for the UGSC-ML sentiment analysis paper.'
    )
    parser.add_argument(
        '--results',
        default='sentiment_classification_results_1875.csv',
        help='Path to the 1,875-row sentiment classification results CSV.'
    )
    parser.add_argument(
        '--lowconf',
        default='low_confidence_annotated_113.csv',
        help='Path to the 113-row annotated low-confidence cases CSV.'
    )
    parser.add_argument(
        '--outdir',
        default='figures',
        help='Output directory for generated figures (created if absent).'
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print('\nLoading data...')
    results_rows = load_results(args.results)
    lc_rows      = load_lowconf(args.lowconf)

    print('\nGenerating Figure 1 — Overall sentiment distribution (donut)...')
    plot_figure1(results_rows, os.path.join(args.outdir, 'figure1_donut.png'))

    print('\nGenerating Figure 2 — Sentiment distribution per language (stacked bar)...')
    plot_figure2(results_rows, os.path.join(args.outdir, 'figure2_stacked_bar.png'))

    print('\nGenerating Figure 3 — Low-confidence taxonomy (bar + dot)...')
    plot_figure3(lc_rows, os.path.join(args.outdir, 'figure3_combined.png'))

    print(f'\nDone. All figures saved to ./{args.outdir}/')


if __name__ == '__main__':
    main()
