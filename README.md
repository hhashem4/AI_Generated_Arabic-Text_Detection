# AI-Generated Arabic Text Detection

This project focuses on **detecting AI-generated Arabic text** using machine learning and deep learning methods. The goal is to classify text as either human-written or AI-generated.

---

##  Dataset

The dataset combines:

- **Human-written abstracts**: Extracted from verified sources  
- **AI-generated abstracts**: Produced using multiple language models for each human-written sample  

**Total samples**: 41,940  

**Features:**

- `abstract_text` – Raw abstract text  
- `source_split` – Data source identifier  
- `generated_by` – Human or AI model name  
- `label` – 1 for human, 0 for AI  

**Engineered Features:**

| Feature | Description |
|---------|-------------|
| F3      | Digits-to-Characters Ratio |
| F26     | Number of Commas |
| F49     | Number of Arabic Particles |
| F72     | Count of Third-Person Pronouns |
| F95     | Polarity Shift Frequency |

---

##  Preprocessing

The preprocessing pipeline includes:

- Arabic text normalization (diacritics removal, orthographic standardization)  
- Filtering non-Arabic characters  
- Tokenization using regex-based Arabic tokenizer  
- Stopword removal (NLTK Arabic stopwords)  
- Stemming using ISRI Stemmer  
- Reconstruction of clean, normalized text  

---

##  Models

### Traditional Machine Learning

| Model                | Validation Accuracy |
|---------------------|------------------|
| Logistic Regression  | 0.9620           |
| SVM                  | 0.9752           |
| Random Forest        | 0.9781           |
| XGBoost              | 0.9692           |

### Deep Learning

- **Architecture**: Dense neural network with two hidden layers (256 → 128 units), dropout 0.3, sigmoid output  
- **Parameters**: 131,585 trainable  
- **Optimizer / Loss**: Adam, Binary Cross-Entropy  
- **Epochs**: 10  

**Test Set Performance:**

| Metric     | Class 0 (Human) | Class 1 (AI) | Macro Avg | Weighted Avg |
|-----------|----------------|---------------|-----------|--------------|
| Precision | 0.89           | 0.76          | 0.83      | 0.87         |
| Recall    | 0.95           | 0.57          | 0.76      | 0.87         |
| F1-score  | 0.92           | 0.65          | 0.79      | 0.87         |
| Support   | 4978           | 1313          | 6291      | 6291         |
| Accuracy  | **0.8724**     |               |           |              |

**Observations:**

- Deep learning performs well on human-written text but struggles with AI-generated text.  
- Traditional ML classifiers, especially Random Forest and SVM, achieve higher overall accuracy and balanced F1-scores.  

---

