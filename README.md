# UGSC-ML: Multilingual Sentiment Analysis for Sustainable Urban Mobility

Repository for the paper:

> **Cross-Lingual Sentiment Classification in Sustainable Mobility: A Zero-Shot Domain Transfer Evaluation Framework**  
> Ainhoa Serna, Jon Kepa Gerrikagoitia, Juan de Oña  
> *AI* (MDPI), 2026. https://doi.org/10.3390/xxxxx

---

## Dataset

The UGSC-ML dataset and model predictions are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20200844.svg)](https://doi.org/10.5281/zenodo.20200844)

Download the following files and place them in this directory:
- `ugsc_ml_translations.csv` — 375 English sentences with aligned translations (ES, FR, DE, IT)
- `sentiment_classification_results_1875.csv` — 1,875 model predictions with confidence scores
- `low_confidence_annotated_113.csv` — 113 annotated low-confidence cases (qualitative taxonomy)

---

## Quickstart

```bash
# 1. Clone and install dependencies
git clone https://github.com/ainhoaserna/UGSC-multilingual-sentiment
cd UGSC-multilingual-sentiment
pip install -r requirements.txt

# 2. Run inference (downloads model ~1 GB on first run)
python inference_pipeline.py \
    --input  ugsc_ml_translations.csv \
    --output sentiment_classification_results.csv

# 3. Generate all figures from the paper
python generate_figures.py \
    --results sentiment_classification_results_1875.csv \
    --lowconf low_confidence_annotated_113.csv \
    --outdir  figures/
```

---

## Scripts

| Script | Description |
|---|---|
| `inference_pipeline.py` | Runs XLM-RoBERTa zero-shot inference on the 5-language dataset (Algorithm 1 from paper) |
| `generate_figures.py` | Generates Figures 1, 2 and 3 from the paper |

### inference_pipeline.py

```
usage: inference_pipeline.py [-h] [--input INPUT] [--output OUTPUT]
                              [--threshold THRESHOLD] [--batch-size BATCH_SIZE]
                              [--device {cpu,cuda,mps}]

options:
  --input      Wide-format CSV with sentence_id, English, Spanish, French, German, Italian
  --output     Output CSV path
  --threshold  Confidence threshold for low-confidence flagging (default: 0.5)
  --batch-size Inference batch size — reduce if out of memory (default: 32)
  --device     cpu | cuda | mps  (default: cpu)
```

### generate_figures.py

```
usage: generate_figures.py [-h] [--results RESULTS] [--lowconf LOWCONF]
                            [--outdir OUTDIR]

options:
  --results  1,875-row sentiment results CSV
  --lowconf  113-row annotated low-confidence cases CSV
  --outdir   Output directory for PNG figures (default: figures/)
```

---

## Model

**CardiffNLP twitter-xlm-roberta-base-sentiment**  
https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment

Applied in zero-shot domain transfer setting: pre-trained on CC100 and fine-tuned on Twitter sentiment data, then applied without task- or domain-specific adaptation to transport reviews.

---

## Citation

```bibtex
@article{serna2026crosslingual,
  title   = {Cross-Lingual Sentiment Classification in Sustainable Mobility:
             A Zero-Shot Domain Transfer Evaluation Framework},
  author  = {Serna, Ainhoa and Gerrikagoitia, Jon Kepa and de O{\~n}a, Juan},
  journal = {AI},
  volume  = {7},
  year    = {2026},
  doi     = {10.3390/xxxxx}
}
```

---

## License

Dataset and code are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
