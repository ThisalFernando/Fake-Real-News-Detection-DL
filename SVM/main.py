import os
from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_svm
from src.evaluate_model import evaluate_model
from src.predict import predict_text

DATA_PATH = "data/news.csv"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    # Preprocess and split into train/validation/test (80/10/10)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, vectorizer = (
        preprocess_data(df)
    )

    print("Training SVM model...")
    # Train SVM and get validation loss
    (
        model,
        scaler,
        val_loss,
        (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test),
    ) = train_svm(X_train_scaled, y_train)

    print("Evaluating model on test set...")
    metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Print summary results
    print("\n-------------------------------------------------")
    print(f"Training Accuracy : {metrics['train_accuracy']:.4f}")
    print(f"Validation Loss   : {val_loss:.4f}")
    print(f"Testing Accuracy  : {metrics['test_accuracy']:.4f}")
    print(f"Overall Accuracy  : {metrics['overall_accuracy']:.4f}")
    print(f"Precision         : {metrics['precision']:.4f}")
    print(f"F1 Score          : {metrics['f1_score']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC           : {metrics['roc_auc']:.4f}")
    print("-------------------------------------------------")

    # Interactive Prediction
    while True:
        print("\nEnter a news article to test (or type 'exit' to quit):")
        user_input = input("> ")

        if user_input.lower() == "exit":
            print("Exiting program...")
            break

        result = predict_text(user_input, vectorizer=vectorizer)
        print(f"Prediction: This news is *{result.upper()}*")


if __name__ == "__main__":
    main()
