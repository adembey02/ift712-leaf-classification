from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class XGBoostModel:
    def __init__(self, param_grid=None, cv=5):
        """
        XGBoost model wrapper with:
        - GPU support if available
        - optional hyperparameter tuning (GridSearchCV)
        - training and evaluation helpers
        """

        # Try GPU first
        try:
            model = XGBClassifier(
                eval_metric="mlogloss",
                n_estimators=1000,          # large, can be reduced by early stopping / tuning
                tree_method="gpu_hist",
                predictor="gpu_predictor",
                device="cuda",
                verbosity=0,
                # regularization / complexity
                max_depth=4,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.5,
                reg_lambda=1.0,
            )
        except Exception:
            print("GPU not available, falling back to CPU")
            model = XGBClassifier(
                eval_metric="mlogloss",
                n_estimators=1000,
                tree_method="hist",
                predictor="cpu_predictor",
                device="cpu",
                verbosity=0,
                max_depth=4,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.5,
                reg_lambda=1.0,
            )

        # Default param_grid if none provided
        if param_grid is None:
            param_grid = {
                "n_estimators": [200, 400, 1000],
                "learning_rate": [0.1],
                "max_depth": [12],
                "gamma": [0, 0.1],
                "reg_lambda": [0, 1],
            }

        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.best_model = model  # updated after tuning

    def tune_hyperparameters(self, X_train, y_train):
        """
        Perform GridSearchCV to find best hyperparameters.
        Updates self.best_model.
        """
        if self.param_grid:
            print("Starting hyperparameter tuning (GridSearchCV)...")
            grid_search = GridSearchCV(
                self.model,
                self.param_grid,
                cv=self.cv,
                scoring="accuracy",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            self.best_model = grid_search.best_estimator_
            print(f"Best parameters found: {grid_search.best_params_}")
        else:
            print("No param_grid provided, skipping tuning.")
            self.best_model = self.model

    def train(self, X_train, y_train):
        """
        Train the (best) model on training data.
        """
        if self.best_model is None:
            self.best_model = self.model

        print("Training XGBoost model...")
        self.best_model.fit(X_train, y_train)
        train_accuracy = self.best_model.score(X_train, y_train)
        print(f"Training Accuracy: {train_accuracy:.4f}")

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data with:
        - accuracy, precision, recall, f1
        - AUC-ROC (binary or multiclass if possible)
        - confusion matrix

        Returns a dict of metrics.
        """
        if self.best_model is None:
            raise RuntimeError("Model has not been trained or tuned yet.")

        y_pred = self.best_model.predict(X_test)

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # AUC-ROC if proba available
        if hasattr(self.best_model, "predict_proba"):
            try:
                y_probs = self.best_model.predict_proba(X_test)
                if y_probs.shape[1] > 2:
                    auc_roc = roc_auc_score(y_test, y_probs, multi_class="ovr")
                else:
                    auc_roc = roc_auc_score(y_test, y_probs[:, 1])
            except Exception as e:
                print(f"Error during probability prediction: {e}")
                auc_roc = "N/A (error during predict_proba)"
        else:
            auc_roc = "N/A (model does not support probability predictions)"

        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "confusion_matrix": conf_matrix,
        }
