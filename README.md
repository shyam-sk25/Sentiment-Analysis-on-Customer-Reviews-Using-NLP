# 🏆 Sentiment Analysis on Amazon Customer Reviews using DistilBERT

This project implements a multi-model approach to classify Amazon Fine Food Reviews into **Positive, Negative, and Neutral** sentiments, demonstrating the power of Transfer Learning (DistilBERT) over traditional methods (Logistic Regression and LSTM) in tackling severe class imbalance.

## 🎯 Problem Statement

[cite_start]The goal was to build an NLP model to accurately analyze customer reviews and predict sentiment[cite: 3]. A critical challenge was the **severe class imbalance**, particularly the low number of 'Neutral' reviews, which caused baseline deep learning models to fail.

## 🚀 Approach and Model Performance

[cite_start]The project followed a defined approach[cite: 10]:

1.  [cite_start]**Data Preprocessing:** Cleaned text, removed stop words, and performed lemmatization[cite: 14, 15].
2.  [cite_start]**Baseline Model (Logistic Regression):** Used TF-IDF features[cite: 18].
3.  [cite_start]**Advanced Model (DistilBERT):** Fine-tuned a pre-trained Transformer model[cite: 24].

| Model | Technique | Overall Accuracy | Neutral F1-Score (Critical Metric) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | TF-IDF | 86.44% | 0.27 |
| **DistilBERT (Final)** | Transfer Learning | **91.04%** | **0.57** |

[cite_start]**Conclusion:** The DistilBERT model achieved the highest performance and successfully corrected the failure to classify the minority 'Neutral' class, validating the use of **Transfer Learning** for robust sentiment analysis[cite: 41].

## 🛠️ Project Deliverables & Files

| File | Description | [cite_start]Project Deliverable [cite: 55, 56, 57] |
| :--- | :--- | :--- |
| `Sentiment Analysis.ipynb` | The complete, executable code including data cleaning, TF-IDF, and DistilBERT fine-tuning. **This is the core script.** | Data Preprocessing, Feature Extraction, Model Training Script |
| `requirements.txt` | Lists all necessary Python libraries for environment setup. | [cite_start]Code Documentation/Reproducibility  |
| `README.md` | Provides the project summary, results, and instructions. | [cite_start]Evaluation Report [cite: 59] |

## ⚙️ Setup and Run Instructions

This project requires a GPU (Tesla T4 recommended) for the DistilBERT fine-tuning step.

### 1. Environment Setup

```bash
# 1. Clone the repository
git clone <your-repo-link>
cd <your-repo-name>

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt
