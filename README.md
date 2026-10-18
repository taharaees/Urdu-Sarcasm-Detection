# Urdu Sarcasm Detection

A natural language processing (NLP) project for binary sarcasm classification of Urdu social-media text. This project compares classical machine learning, a BiLSTM neural network, and a fine-tuned XLM-RoBERTa transformer.

## Project Overview

Sarcasm detection is a challenging NLP task because sarcastic statements can resemble ordinary jokes, teasing, criticism, and literal statements. This project investigates how different text representations and model architectures perform on Urdu social-media text.

The experiments include:

- Word- and character-level TF-IDF features with Logistic Regression.
- A Keras Bidirectional Long Short-Term Memory (BiLSTM) classifier.
- Fine-tuning the `xlm-roberta-base` transformer for binary classification.
- An ablation study comparing word, character, and combined TF-IDF features.
- An emoji-removal experiment.
- Validation-based model selection, held-out test evaluation, and error analysis.

## Dataset

The dataset contains Urdu social-media text labeled for sarcasm.

| Property               | Value                       |
| ---------------------- | --------------------------- |
| Original dataset size  | 20,060 rows                 |
| Final modeling dataset | 15,216 unique labeled texts |
| Non-sarcastic class    | 9,037 (59.39%)              |
| Sarcastic class        | 6,179 (40.61%)              |
| Training split         | 10,650 samples              |
| Validation split       | 2,283 samples               |
| Test split             | 2,283 samples               |
| Random seed            | 42                          |

The target labels are:

- `0` — Non-sarcastic
- `1` — Sarcastic

Data preparation includes missing-value and label checks, removal of texts with conflicting labels, deduplication, Urdu-aware Unicode normalization, and stratified train/validation/test splitting.

Emoji, punctuation, hashtags, mentions, and Roman Urdu or code-mixed text are preserved in the primary preprocessing pipeline.

**Dataset note:** Before redistributing the CSV publicly, verify that you have permission to share the original dataset. If redistribution is not permitted, exclude the dataset file and document how authorized users can obtain it.

## Models and Experiments

### 1. TF-IDF + Logistic Regression

The classical baseline combines word unigram/bigram TF-IDF features and character 2–5-gram TF-IDF features. Logistic Regression is trained with balanced class weights.

### 2. BiLSTM

A neural sequence classifier built with Keras, using an embedding layer, bidirectional LSTM, dense layer, dropout, and a sigmoid output.

### 3. XLM-RoBERTa

The multilingual `xlm-roberta-base` transformer is fine-tuned for two-class classification. Training used a maximum sequence length of 96 tokens, a learning rate of `2e-5`, gradient accumulation, mixed precision, and gradient checkpointing.

## Results

The primary model-selection metric is **Macro-F1**.

### Validation Results

| Model                               | Accuracy | Macro-F1 |
| ----------------------------------- | -------: | -------: |
| TF-IDF word + character, emoji kept |   0.8327 |   0.8264 |
| BiLSTM                              |   0.8025 |   0.7984 |
| XLM-RoBERTa                         |   0.8406 |   0.8366 |

XLM-RoBERTa was selected using validation Macro-F1.

### Held-Out Test Results

| Model                                         | Accuracy | Macro-F1 |
| --------------------------------------------- | -------: | -------: |
| XLM-RoBERTa                                   |   0.8200 |   0.8159 |
| TF-IDF word + character / Logistic Regression |   0.8195 |   0.8139 |
| BiLSTM                                        |   0.8108 |   0.8074 |

The fine-tuned XLM-RoBERTa model achieved the best held-out test Macro-F1. However, the TF-IDF baseline performed nearly as well while requiring a simpler model architecture.

These results are specific to the dataset, preprocessing, split, and training configuration used in this project.

## Key Findings

- **Transformer performance:** XLM-RoBERTa achieved a validation Macro-F1 of 0.8366 and a held-out test Macro-F1 of 0.8159.
- **Strong classical baseline:** Word- and character-level TF-IDF features with Logistic Regression achieved a test Macro-F1 of 0.8139.
- **Character-level features matter:** Character n-grams outperformed word n-grams alone in the reported ablation study.
- **Emoji removal had little effect:** Removing emoji produced only a small change in validation performance.
- **BiLSTM overfitting:** The BiLSTM showed a widening gap between training and validation performance.
- **Sarcasm remains context-dependent:** Humor, teasing, political criticism, implicit irony, and inconsistent annotations contribute to classification errors.

## Error Analysis

On the held-out test set, the selected XLM-RoBERTa model misclassified 411 of 2,283 examples, approximately 18.0%.

The main observed error patterns include:

- Humorous but non-sarcastic statements predicted as sarcastic.
- Implicit or context-dependent sarcasm missed by the model.
- Political criticism incorrectly associated with sarcasm.
- Ambiguity between friendly teasing, jokes, mockery, and irony.

## Environment

The experiments were run in Google Colab using Python 3.13 and a Tesla T4 GPU for transformer fine-tuning.

The original notebook installs dependencies including TensorFlow, Transformers, Accelerate, and Datasets. Other libraries used by the experiments include PyTorch, scikit-learn, pandas, NumPy, and Matplotlib.

## How to Use

1. Clone or download this repository.
2. Install the dependencies in a suitable Python environment.
3. Open the notebook in Jupyter or Google Colab.
4. Configure the dataset path or upload the CSV when prompted by the notebook.
5. Run the cells in order to reproduce the data preparation, training, evaluation, and error-analysis workflow.

Transformer training is computationally intensive. A GPU-enabled environment is recommended for fine-tuning XLM-RoBERTa.

If the trained model artifacts are included, they can be used for inference without repeating the full training process, provided the required dependencies and preprocessing steps are available.

## Fine-tuned XLM-RoBERTa Model

The trained model weights and tokenizer are hosted on Hugging Face.

[View the model on Hugging Face](https://huggingface.co/taharaees/urdu-sarcasm-xlmr/tree/main)

## Reproducibility

- Random seed: `42`
- Data splitting: stratified 70/15/15 train/validation/test
- Primary metric: Macro-F1
- Model selection: best validation Macro-F1
- Final evaluation: held-out test set

## Limitations and Future Work

Sarcasm detection depends on linguistic, cultural, and conversational context. Dataset annotation inconsistencies and overlap between humor and sarcasm limit model reliability.

Potential improvements include collecting cleaner labeled data, adding conversational context, tuning the classification threshold, and manually reviewing ambiguous humor–sarcasm examples.

