from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hinge_loss
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import os


def train_svm(X, y, model_path="model/svm_model.pkl", quick_tune=True):

    print("Splitting data (80/10/10)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Define hyperparameter grid
    if quick_tune is True:
        param_grid = {
            "C": [0.001, 0.005, 0.01, 0.05, 0.1],  # Regularization strength
            "class_weight": [None, "balanced"],  # Handles class imbalance
            "max_iter": [10000],  # Iterations for convergence
        }
    else:
        param_grid = {
            "C": [0.001, 0.01, 0.05, 0.1, 1],
            "class_weight": [None, "balanced"],
            "max_iter": [5000, 10000],
            "dual": [False],
            "loss": ["squared_hinge"],  # SVM loss function
        }

    # Hyperparameter tuning
    print("Running Grid Search (safe mode, n_jobs=1)...")
    grid_search = GridSearchCV(
        LinearSVC(),
        param_grid,
        cv=3,  # 3-fold cross-validation
        scoring="accuracy",
        verbose=2,
        n_jobs=1,  # Avoid potential issues with multi-threading
    )

    # Fit model
    grid_search.fit(X_train_scaled, y_train)

    # Create results directory for plots
    results_dir = "../results" if os.path.isdir("../results") else "results"
    os.makedirs(results_dir, exist_ok=True)

    # Get best model from grid search
    best_svm = grid_search.best_estimator_
    print("\nBest Hyperparameters:", grid_search.best_params_)
    print("Best Cross-validation Score:", grid_search.best_score_)

    # Plot - Model Accuracy vs Regularization (C)
    try:
        # Extract mean test scores for different C (grid may include other params)
        results = grid_search.cv_results_
        params = results["params"]
        mean_test_scores = results["mean_test_score"]

        # Collect scores by C value (take max score for that C across other params)
        c_scores = {}
        for p, score in zip(params, mean_test_scores):
            c_val = p.get("C")
            if c_val is None:
                continue
            c_scores.setdefault(c_val, []).append(score)

        c_values = sorted(c_scores.keys())
        c_means = [np.mean(c_scores[c]) for c in c_values]

        plt.figure()
        plt.semilogx(c_values, c_means, marker="o")
        plt.xlabel("Regularization (C)")
        plt.ylabel("Cross-validated Accuracy")
        plt.title("Model Accuracy vs Regularization (C)")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.savefig(os.path.join(results_dir, "accuracy_vs_C.png"))
        plt.close()
        print(f"Saved Accuracy vs C plot to {results_dir}/accuracy_vs_C.png")
    except Exception as e:
        print(f"Could not create Accuracy vs C plot: {e}")

    # Plot - Learning Curve (Accuracy vs Training Set Size)
    try:
        # Use learning_curve on the best estimator with the scaled full training data
        train_sizes, train_scores, test_scores = learning_curve(
            best_svm,
            X_train_scaled,
            y_train,
            cv=3,
            scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 5),
            n_jobs=1,
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        plt.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )
        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve (Accuracy vs Training Set Size)")
        plt.legend(loc="best")
        plt.grid(True, ls="--", lw=0.5)
        plt.savefig(os.path.join(results_dir, "learning_curve_accuracy.png"))
        plt.close()
        print(f"Saved Learning Curve plot to {results_dir}/learning_curve_accuracy.png")
    except Exception as e:
        print(f"Could not create Learning Curve plot: {e}")

    # Evaluate validation loss
    y_val_num = [0 if label == "fake" else 1 for label in y_val]  # convert to numeric
    y_val_decision = best_svm.decision_function(X_val_scaled)
    val_loss = hinge_loss(y_val_num, y_val_decision)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save model and scaler
    dump((best_svm, scaler), model_path)
    print(f"Model and scaler saved to {model_path}")

    # Return model, scaler, val_loss, and all splits
    return (
        best_svm,
        scaler,
        val_loss,
        (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test),
    )
