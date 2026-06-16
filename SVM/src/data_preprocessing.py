import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(path):
    """Load dataset from CSV file"""
    return pd.read_csv(path)


def preprocess_data(df):
    """Preprocess text and split into train/validation/test sets"""
    # Drop rows where text or label is missing
    df = df.dropna(subset=["text", "label"])

    # Convert labels to lowercase
    df["label"] = df["label"].str.lower()

    # Keep only 'real' and 'fake' labels
    df = df[df["label"].isin(["real", "fake"])]

    X = df["text"]
    y = df["label"]

    # First split: 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: 50% of temp -> validation, 50% -> test (each 10% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf, y_test, vectorizer
