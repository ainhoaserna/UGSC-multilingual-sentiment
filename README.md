# UGSC-multilingual-sentiment
Multilingual version of the User Gold Standard Corpus (UGSC) for evaluating cross-lingual sentiment analysis in sustainable urban mobility.
# UGSC Multilingual Sentiment Dataset for Sustainable Mobility

This repository contains the multilingual version of the **User Gold Standard Corpus (UGSC)**, designed for evaluating cross-lingual sentiment classification models in the context of sustainable urban mobility.

## Description

The dataset supports research on **zero-shot multilingual sentiment analysis** using real-world user-generated content (UGC) from platforms like TripAdvisor. It is based on an English corpus manually annotated with sentiment polarity labels and includes sentence-aligned translations in:

- 🇪🇸 Spanish
- 🇫🇷 French
- 🇩🇪 German
- 🇮🇹 Italian

## Contents
- `translated_versions/` — Sentence-aligned corpora in five languages
- `sentiment_predictions/` — Model outputs using XLM-RoBERTa (label + confidence)
- `visualizations/` — Distribution plots and analysis figures
- `README.md` — Description and instructions
- `LICENSE` — Open data license (recommended: CC-BY-4.0)
  
## Keywords

`Multilingual NLP`, `Sentiment Analysis`, `Sustainable Mobility`, `Zero-Shot Learning`, `Cross-Lingual Evaluation`, `User-Generated Content`, `NLP for Transport`

## How to use

1. Load any of the translated corpora from `translated_versions/`.
2. Run inference using the `cardiffnlp/twitter-xlm-roberta-base-sentiment` model via HuggingFace.
3. Compare predictions and confidence across languages.
4. Use figures and plots for exploratory or benchmarking purposes.
   
## Citation

If you use this dataset or findings from the associated publication, please cite:

Serna, A. (2025). *Cross-Lingual Sentiment Classification in Sustainable Mobility: A Zero-Shot Evaluation Framework*.


## License

MIT License / CC BY-SA 4.0

