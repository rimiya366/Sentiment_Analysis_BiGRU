# 🎭 Sentiment Analysis Engine (BiGRU + Keras)

A deep learning sentiment classifier built using **TensorFlow/Keras** and **Bidirectional GRU** networks to perform binary sentiment analysis (Positive/Negative) on the IMDB movie reviews dataset.

---

## 📌 Project Overview

This repository contains an end-to-end NLP pipeline for text classification. The architecture uses word embeddings, bidirectional recurrent units (BiGRU), L2 regularization, and dynamic training callbacks (`EarlyStopping` and `ReduceLROnPlateau`) to prevent overfitting and ensure robust generalization.

### Key Features
* **Word Embeddings:** Converts raw tokens into dense 128-dimensional vectors.
* **Bidirectional GRU:** Captures contextual information from both left-to-right and right-to-left sequence passes.
* **Regularization Pipeline:** Utilizes input/recurrent dropout alongside L2 weight penalties to control overfitting.
* **Optimized Preprocessing:** Employs `padding='pre'` to eliminate RNN hidden-state memory decay over zero-padded tokens.
* **Inference Pipeline:** Includes custom text pre-processing and probability classification logic.

---

## 🏗️ Model Architecture
[ Input Tokens (200,) ]
│
[ Embedding Layer (10000 -> 128) ]
│
[ Bidirectional GRU (64 units) ]
│
[ Dense Layer (32 units, ReLU + L2) ]
│
[ Dropout Layer (0.5) ]
│
[ Dense Output (1 unit, Sigmoid + L2) ]

| Layer | Output Shape | Parameters | Purpose |
| :--- | :--- | :--- | :--- |
| **Input** | `(None, 200)` | 0 | Sequence length capped at 200 tokens |
| **Embedding** | `(None, 200, 128)` | 1,280,000 | Maps vocab indices to 128D space |
| **Bidirectional GRU** | `(None, 128)` | 74,496 | Extracts temporal features across directions |
| **Dense (Hidden)** | `(None, 32)` | 4,128 | Non-linear feature combination with L2 penalty |
| **Dropout** | `(None, 32)` | 0 | Regularization (50% drop rate) |
| **Dense (Output)** | `(None, 1)` | 33 | Outputs probability score $[0.0, 1.0]$ |

---

## 📊 Training & Performance Results

* **Validation Accuracy:** ~83.5%
* **Validation Loss:** ~0.444
* **Callbacks:**
  * `EarlyStopping`: Halts training automatically when validation loss plateaus to preserve optimal weights.
  * `ReduceLROnPlateau`: Scales down learning rate by factor $0.5$ when loss hits a local minimum.

---

## 🚀 Quick Start

### Prerequisites
* Python 3.9+
* TensorFlow 2.x
* NumPy

### Installation
```bash
git clone [https://github.com/rimiya366/sentiment-analysis-bigru.git](https://github.com/rimiya366/sentiment-analysis-bigru.git)
cd sentiment-analysis-bigru
pip install tensorflow numpy
