import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(
    model, X_train, y_train, X_test, y_test, model_name="SVM", results_dir="results"
):
    """Evaluate model performance, including Accuracy, Precision, F1, ROC-AUC, and confusion matrices."""

    os.makedirs(results_dir, exist_ok=True)

    # Training Predictions
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_acc:.4f}")

    # Testing Predictions
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Testing Accuracy: {test_acc:.4f}")

    # Overall Accuracy
    y_all_true = np.concatenate([y_train, y_test])
    y_all_pred = np.concatenate([y_train_pred, y_test_pred])
    overall_acc = accuracy_score(y_all_true, y_all_pred)
    print(f"Overall Accuracy: {overall_acc:.4f}")

    # Convert string labels to numeric for metrics
    label_mapping = {"fake": 0, "real": 1}
    y_test_num = np.array([label_mapping[y] for y in y_test])
    y_test_pred_num = np.array([label_mapping[y] for y in y_test_pred])

    # Precision, F1, ROC-AUC
    precision = precision_score(y_test_num, y_test_pred_num)
    f1 = f1_score(y_test_num, y_test_pred_num)

    try:
        y_scores = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test_num, y_scores)
    except Exception as e:
        roc_auc = None
        print(f"ROC-AUC could not be calculated: {e}")

    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")

    # Save metrics to CSV file
    report = classification_report(y_test, y_test_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df["train_accuracy"] = train_acc
    metrics_df["test_accuracy"] = test_acc
    metrics_df["overall_accuracy"] = overall_acc
    metrics_df["precision"] = precision
    metrics_df["f1_score"] = f1
    metrics_df["roc_auc"] = roc_auc

    metrics_path = os.path.join(results_dir, f"{model_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=True)
    print(f"Metrics saved to {metrics_path}")

    # Train Confusion Matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_train,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Train) - {model_name}")
    plt.savefig(os.path.join(results_dir, f"{model_name}_confusion_matrix_train.png"))
    plt.close()

    # Normalized Train Confusion Matrix
    cm_train_norm = confusion_matrix(y_train, y_train_pred, normalize="true")
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_train_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Normalized Confusion Matrix (Train) - {model_name}")
    plt.savefig(
        os.path.join(results_dir, f"{model_name}_confusion_matrix_train_normalized.png")
    )
    plt.close()

    # Test Confusion Matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_test,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Test) - {model_name}")
    plt.savefig(os.path.join(results_dir, f"{model_name}_confusion_matrix_test.png"))
    plt.close()

    # Normalized Test Confusion Matrix
    cm_test_norm = confusion_matrix(y_test, y_test_pred, normalize="true")
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_test_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Normalized Confusion Matrix (Test) - {model_name}")
    plt.savefig(
        os.path.join(results_dir, f"{model_name}_confusion_matrix_test_normalized.png")
    )
    plt.close()

    # ROC Curve Plot
    if roc_auc is not None:
        fpr, tpr, _ = roc_curve(y_test_num, y_scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(results_dir, f"{model_name}_roc_curve.png"))
        plt.close()

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "overall_accuracy": overall_acc,
        "precision": precision,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }
