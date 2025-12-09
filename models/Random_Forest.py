from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import numpy as np


class RandomForestModel:
    def __init__(self, config, cv=5):
        """
        Fully self-contained RandomForest model wrapper.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing RF hyperparameters lists.
        cv : int
            Number of folds for GridSearchCV.
        """

        rf_cfg = config["models"]["random_forest"]

        # Base model using the first values
        self.model = RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"][0],
            max_depth=rf_cfg["max_depth"][0],
            criterion=rf_cfg["criterion"][0],
            random_state=42
        )

        # Grid for tuning
        self.param_grid = {
            "n_estimators": rf_cfg["n_estimators"],
            "max_depth": rf_cfg["max_depth"],
            "criterion": rf_cfg["criterion"],
        }

        self.cv = cv
        self.best_model = None

    #################################################################
    # TRAINING
    #################################################################
    def train(self, X_train, y_train, tune=True):
        """
        Train RF classifier (optionally with hyperparameter tuning).
        """
        if tune:
            self.tune_hyperparameters(X_train, y_train)
        else:
            self.best_model = self.model

        print("Training RandomForest model...")
        self.best_model.fit(X_train, y_train)
        print("Training accuracy:", self.best_model.score(X_train, y_train))

    #################################################################
    # GRID SEARCH
    #################################################################
    def tune_hyperparameters(self, X_train, y_train):
        """
        Run GridSearchCV and update self.best_model.
        """
        print("Starting RandomForest hyperparameter tuning...")

        grid = GridSearchCV(
            self.model,
            self.param_grid,
            cv=self.cv,
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        self.best_model = grid.best_estimator_
        print("Best parameters:", grid.best_params_)

    #################################################################
    # EVALUATION
    #################################################################
    def evaluate(self, X_test, y_test):
        """
        Evaluate RF on multiple metrics and return a dict.
        """
        if self.best_model is None:
            raise RuntimeError("Model must be trained before evaluation.")

        y_pred = self.best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Try AUC if model supports predict_proba
        if hasattr(self.best_model, "predict_proba"):
            try:
                y_prob = self.best_model.predict_proba(X_test)
                if y_prob.shape[1] > 2:
                    auc_roc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                else:
                    auc_roc = roc_auc_score(y_test, y_prob[:, 1])
            except:
                auc_roc = "N/A"
        else:
            auc_roc = "N/A"

        conf_mat = confusion_matrix(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("AUC:", auc_roc)
        print("Confusion matrix:\n", conf_mat)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
            "confusion_matrix": conf_mat,
        }

    #################################################################
    # PREDICTION
    #################################################################
    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Model must be trained before predictions.")
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None:
            raise RuntimeError("Model must be trained before predictions.")
        return self.best_model.predict_proba(X)
