from joblib import load


def predict_text(text, model_path="model/svm_model.pkl", vectorizer=None):
    """Predict if the news article is real or fake"""
    # Load the model and scaler
    model, scaler = load(model_path)

    # Transform text using the vectorizer
    X = vectorizer.transform([text])

    # Apply the same scaling as training (important for LinearSVC)
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]
    return prediction
