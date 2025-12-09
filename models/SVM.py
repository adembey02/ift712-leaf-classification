from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from utils.split import OneVsOneSplit, OneVsRestSplit
import numpy as np


class SVMModel:
    def __init__(self, config, binaire=False, noyau='rbf', cv=5):
        """
        Fully self-contained SVM classifier model (no BaseModel).

        Parameters:
        -----------
        config : dict
            Configuration dictionary (C, gamma, degree, strategy…)
        binaire : bool
            If True: NonDemented vs others
        noyau : str
            'linear', 'rbf', or 'poly'
        cv : int
            Number of folds for GridSearchCV
        """

        self.binaire = binaire
        self.noyau = noyau
        self.cv = cv
        self.multiclass_strategy = config["classification"]["multiclass_strategy"]

        if noyau not in ["linear", "rbf", "poly"]:
            raise ValueError("Kernel must be 'linear', 'rbf', or 'poly'")

        ##################################################################
        # Model definition (binary or multiclass)
        ##################################################################
        if binaire:
            self.model = SVC(kernel=noyau, probability=True)

            # Parameter grid for GridSearchCV
            if noyau == "linear":
                self.param_grid = {"C": config["models"]["svm"]["C"]}

            elif noyau == "rbf":
                self.param_grid = {
                    "C": config["models"]["svm"]["C"],
                    "gamma": config["models"]["svm"]["gamma"]
                }

            elif noyau == "poly":
                self.param_grid = {
                    "C": config["models"]["svm"]["C"],
                    "degree": config["models"]["svm"]["degree"],
                    "gamma": config["models"]["svm"]["gamma"]
                }

        else:
            # Multiclass wrapping (ovo or ovr)
            base = SVC(kernel=noyau, probability=True)

            if self.multiclass_strategy == "ovo":
                self.model = OneVsOneSplit(base)
            else:  # default = ovr
                self.model = OneVsRestSplit(base)

            # Param grid in sklearn format for meta-estimators
            if noyau == "linear":
                self.param_grid = {
                    "estimator__C": config["models"]["svm"]["C"]
                }

            elif noyau == "rbf":
                self.param_grid = {
                    "estimator__C": config["models"]["svm"]["C"],
                    "estimator__gamma": config["models"]["svm"]["gamma"]
                }

            elif noyau == "poly":
                self.param_grid = {
                    "estimator__C": config["models"]["svm"]["C"],
                    "estimator__degree": config["models"]["svm"]["degree"],
                    "estimator__gamma": config["models"]["svm"]["gamma"]
                }

        self.best_model = None

    ##################################################################
    # LABEL PREPROCESSING
    ##################################################################
    def pretraiter_etiquettes(self, y):
        if self.binaire:
            return np.array(
                ["NonDemented" if v == "NonDemented" else "Demented" for v in y]
            )
        return y

    ##################################################################
    # HYPERPARAMETER TUNING
    ##################################################################
    def tune_hyperparameters(self, X_train, y_train):
        """
        Runs GridSearchCV and sets self.best_model.
        """
        print("Starting SVM hyperparameter tuning...")

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

    ##################################################################
    # TRAINING
    ##################################################################
    def train(self, X_train, y_train, tune=True):
        """
        Train SVM, optionally with hyperparameter tuning.
        """
        y_train_prep = self.pretraiter_etiquettes(y_train)

        if tune:
            self.tune_hyperparameters(X_train, y_train_prep)
        else:
            self.best_model = self.model

        print("Training SVM model...")
        self.best_model.fit(X_train, y_train_prep)
        print("Training accuracy:", self.best_model.score(X_train, y_train_prep))

    ##################################################################
    # EVALUATION
    ##################################################################
    def evaluate(self, X_test, y_test):
        """
        Returns dictionary of metrics.
        """
        if self.best_model is None:
            raise RuntimeError("Model must be trained before evaluation")

        y_test_prep = self.pretraiter_etiquettes(y_test)
        y_pred = self.best_model.predict(X_test)

        # Scores
        accuracy = accuracy_score(y_test_prep, y_pred)
        precision = precision_score(y_test_prep, y_pred, average="weighted")
        recall = recall_score(y_test_prep, y_pred, average="weighted")
        f1 = f1_score(y_test_prep, y_pred, average="weighted")

        # Try AUC-ROC if proba exists
        if hasattr(self.best_model, "predict_proba"):
            try:
                y_probs = self.best_model.predict_proba(X_test)
                if y_probs.shape[1] > 2:
                    auc_roc = roc_auc_score(y_test_prep, y_probs, multi_class="ovr")
                else:
                    auc_roc = roc_auc_score(y_test_prep, y_probs[:, 1])
            except:
                auc_roc = "N/A"

        else:
            auc_roc = "N/A"

        conf_mat = confusion_matrix(y_test_prep, y_pred)

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

    ##################################################################
    # PREDICTION
    ##################################################################
    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Model must be trained first")
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None:
            raise RuntimeError("Model must be trained first")
        return self.best_model.predict_proba(X)
