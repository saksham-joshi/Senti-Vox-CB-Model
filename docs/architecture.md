# SentiVox — Model Architecture & Fine-Tuning Documentation

## 1. Overview

SentiVox is a multilingual sentiment analysis platform that performs **multi-label emotion classification** using Google's GoEmotions taxonomy (28 labels), then derives **document-level sentiment** (positive / negative / neutral / ambiguous) via a deterministic mapping layer.

## 2. Model Architecture

### 2.1 Base Model: MuRIL (google/muril-base-cased)

- **Type**: BERT-based transformer (12 layers, 768 hidden, 12 heads)
- **Parameters**: ~237M
- **Pretraining**: Multilingual Representations for Indian Languages
  - Trained on 17 Indian languages + English
  - Uses transliterated data for cross-script transfer
- **Why MuRIL?**
  - Native support for Indian languages (Hindi, Bengali, Tamil, Telugu, etc.)
  - Strong cross-lingual transfer — fine-tuning on English GoEmotions can generalize to Indian languages
  - Apache 2.0 license — free for commercial use

### 2.2 Classification Head

```
MuRIL Encoder (768-dim)
    │
    ▼
[CLS] Token Pooling
    │
    ▼
Dropout (p=0.1)
    │
    ▼
Linear (768 → 28)  ← Multi-label classification head
    │
    ▼
Sigmoid (per-label)
    │
    ▼
Threshold (≥ 0.5) → Active emotions
    │
    ▼
Emotion-to-Sentiment Mapping → Final sentiment
```

### 2.3 Why Multi-Label (not Multi-Class)?

GoEmotions is inherently multi-label — a single text can express multiple emotions simultaneously (e.g., "I love this but I'm scared" → `love` + `fear`). We use:
- **BCEWithLogitsLoss** (Binary Cross-Entropy with Logits) — standard for multi-label problems
- **Sigmoid activation** per label instead of softmax
- **Threshold-based prediction** (default: 0.5)

## 3. Emotion → Sentiment Mapping

The 28 GoEmotions labels are mapped into 4 sentiment buckets:

| Sentiment | Emotions |
|-----------|----------|
| **Positive** | admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief |
| **Negative** | anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness |
| **Ambiguous** | confusion, curiosity, realization, surprise |
| **Neutral** | neutral |

### Aggregation Rules:
1. If all predicted emotions map to the same bucket → return that bucket
2. If both positive AND negative are present → return **"ambiguous"**
3. Otherwise → majority vote among buckets

## 4. Fine-Tuning Process

### 4.1 Dataset

- **GoEmotions** (Demszky et al., 2020)
- ~58k Reddit comments, manually labeled with 27 emotions + neutral
- Split: 60% train / 20% validation / 20% test

### 4.2 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 2e-5 | Standard for BERT fine-tuning |
| Batch Size | 32 (effective: 64) | Gradient accumulation to fit in Colab T4 |
| Epochs | 5 | With early stopping (patience=2) |
| Max Length | 128 tokens | GoEmotions texts are short (Reddit comments) |
| Warmup | 10% of total steps | Prevents early divergence |
| Weight Decay | 0.01 | L2 regularization |
| FP16 | Enabled | Halves memory, ~2x speedup on T4 |
| Optimizer | AdamW | Transformer standard |

### 4.3 Training Strategy

1. **Full fine-tuning** — all MuRIL parameters + classification head are trainable
2. **Early stopping** on validation `f1_macro` with patience=2 epochs
3. **Best checkpoint** is automatically loaded at the end
4. Only the 2 most recent checkpoints are kept (saves Colab disk)

### 4.4 Evaluation Metrics

- **Accuracy**: Exact match (all 28 labels must match per sample)
- **F1-score (macro)**: Average F1 across all 28 labels — treats rare emotions equally
- **F1-score (micro)**: Global TP/FP/FN — reflects overall performance
- **Precision / Recall**: Both macro and micro variants
- **Confusion Matrix**: Per-emotion 2×2 matrices + aggregated sentiment-level matrix

## 5. Document-Level Sentiment

For longer texts (multiple sentences/paragraphs), SentiVox uses a **segment-aggregate** strategy:

1. Split document into segments (sentences or paragraphs)
2. Predict emotions for each segment independently
3. Take the **union** of all predicted emotions
4. Apply sentiment mapping to the aggregated emotion set

This approach:
- Handles mixed-sentiment documents (e.g., product reviews)
- Avoids truncation issues with BERT's 512-token limit
- Is simple and interpretable

## 6. Output Format

```json
{
    "sentiment": "positive",
    "emotions": ["joy", "gratitude"],
    "confidence": {
        "admiration": 0.12,
        "joy": 0.87,
        "gratitude": 0.71,
        "neutral": 0.05,
        ...
    }
}
```

## 7. Recommendations for Further Improvement

### 7.1 Multilingual Expansion

MuRIL already supports Indian languages. To add Hindi/Tamil/etc.:

1. **Translate GoEmotions** to the target language (e.g., using IndicTrans2 or Google Translate + human validation)
2. **Fine-tune further** on the translated dataset — MuRIL's shared multilingual representations enable strong cross-lingual transfer
3. **Mixed-language training** — combine English + Hindi + Tamil datasets for a single robust model

The codebase is designed for this:
- Just create a new JSON file (e.g., `goemotions-hin.json`) with the same format
- Update the data loading path
- Re-run training

### 7.2 Performance Improvements

- **Class weighting**: GoEmotions has severe class imbalance (e.g., `grief` is rare). Use `pos_weight` in BCEWithLogitsLoss
- **Threshold tuning**: Find optimal per-label thresholds using validation set (instead of fixed 0.5)
- **Longer context**: For document-level, consider a hierarchical approach (sentence embeddings → document classifier)
- **Ensemble**: Combine MuRIL with XLM-RoBERTa for higher accuracy

### 7.3 Deployment

- **HuggingFace Hub**: Model is saved in HF format, ready for `api.upload_folder()`
- **HuggingFace Inference API**: Free hosting for models on the Hub
- **ONNX export**: For faster inference in production (`optimum` library)

## 8. References

1. Demszky et al. (2020). [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547)
2. Khanuja et al. (2021). [MuRIL: Multilingual Representations for Indian Languages](https://arxiv.org/abs/2103.10730)
3. Devlin et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
