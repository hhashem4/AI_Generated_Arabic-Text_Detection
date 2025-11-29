# ---------------------------------------------------------
# data_preparation.py
# Utilities for data cleaning, preprocessing, tokenization,
# and feature engineering
# ---------------------------------------------------------

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

# Download stopwords if not already present
nltk.download("stopwords")

# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def remove_diacritics(text):
    """Remove Arabic diacritics from text."""
    arabic_diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return re.sub(arabic_diacritics, '', text)


def normalize_arabic(text):
    """Normalize Arabic characters to a unified form."""
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[^؀-ۿ ]+", " ", text)  # Remove non-Arabic characters
    return text


# Stopwords + Stemmer
arabic_stopwords = set(stopwords.words("arabic"))
stemmer = ISRIStemmer()


# ---------------------------------------------------------
# PREPROCESSING PIPELINE
# ---------------------------------------------------------

def preprocess_text(text):
    """Full preprocessing: normalization, stopword removal, stemming."""
    text = str(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)

    tokens = text.split()
    tokens = [w for w in tokens if w not in arabic_stopwords]
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------
# TOKENIZATION UTILITIES
# ---------------------------------------------------------

try:
    import re2  # If available for faster regex
except:
    re2 = re

def simple_word_tokenize(text):
    """Tokenize into Arabic words, Latin words, and punctuation."""
    return re2.findall(r"\p{Arabic}+|\w+|[^\s\w]", text, flags=re2.VERSION1)


def sentence_tokenize(text):
    """Split text into sentences using Arabic & English punctuation."""
    parts = re.split(r'(?<=[\.\?\!\u061F\u061B])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def paragraph_tokenize(text):
    """Split text into paragraphs based on blank lines."""
    if not isinstance(text, str):
        return []
    paragraphs = re.split(r'\s*\n\s*\n\s*', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

TATWEEL = "ـ"

def apply_feature_engineering(df):
    """Generate all engineered features and token-level structures."""

    # Tokenization
    df["tokens"] = df["clean_text"].apply(
        lambda t: [tok for tok in simple_word_tokenize(t) if tok.strip()]
    )

    df["words"] = df["tokens"].apply(
        lambda toks: [tok for tok in toks if re.search(r'\w', tok)]
    )

    df["sentences"] = df["clean_text"].apply(sentence_tokenize)
    df["paragraphs"] = df["clean_text"].apply(paragraph_tokenize)



    # F3 — Digits / Characters
    df["f003_digits_over_C"] = df["clean_text"].apply(
        lambda t: len(re.findall(r'\d', str(t))) / len(str(t))
        if len(str(t)) > 0 else 0
    )

    # Feature 26 — Number of commas
    df["f026_commas"] = df["clean_text"].apply(
        lambda t: str(t).count(",") if isinstance(t, str) else 0
    )


    # Feature 49 — Number of Arabic Particles (أدوات)
    
        arabic_particles = {
        'من','إلى','عن','على','في','ب','ك','ل',
        'و','أو','ثم','بل','لكن',
        'لا','لم','لن','ما',
        'هل','إن','إذا','أين','متى','كيف','كم','أيان',
        'قد','لمّا','حتى','أن','إنّ','إذن'
          }

    df["f049_num_particles"] = df["words"].apply(
        lambda words: sum(1 for w in words if w in arabic_particles)
        if isinstance(words, list) else 0
    )


    # Feature 72 — Third Person Pronouns (proxy for grammatical person)
    
    third_person_pronouns = {
        "هو","هي","هم","هما","هن",
        "ذلك","تلك","ذلكم","ذلكما","تلكم"
    }

    df["f072_third_person_pronouns"] = df["words"].apply(
        lambda words: sum(1 for w in words if w in third_person_pronouns)
        if isinstance(words, list) else 0
    )


    # Feature 95 — Polarity Shift Frequency Across Sentences
   

    pos_words = ["جيد","مفيد","تحسين","إيجابي","نجاح","فعالية"]
    neg_words = ["سيء","ضعيف","سلبي","فشل","خسارة","مشكلة"]

    def polarity_of(sentence):
        score = 0
        if any(w in sentence for w in pos_words): score += 1
        if any(w in sentence for w in neg_words): score -= 1
        return 1 if score > 0 else -1 if score < 0 else 0

    df["f095_polarity_shift"] = df["sentences"].apply(
        lambda S: sum(
            1 for a, b in zip(
                [polarity_of(s) for s in S][:-1],
                [polarity_of(s) for s in S][1:]
            ) if a != b
        ) if isinstance(S, list) and len(S) > 1 else 0
    )
   

    return df
