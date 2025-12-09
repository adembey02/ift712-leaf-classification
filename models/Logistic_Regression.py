from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from utils.split import OneVsOneSplit, OneVsRestSplit
import numpy as np


class LogisticRegressionModel:
    def __init__(self, config, binaire=False, cv=5):
        """
        Modèle de régression logistique auto-contenu (sans BaseModel).

        Paramètres
        ----------
        config : dict
            Dictionnaire de configuration du modèle.
        binaire : bool
            Si True : classification binaire.
            Si False : classification multiclasse (OVO/OVR).
        cv : int
            Nombre de folds pour GridSearchCV.
        """
        self.binaire = binaire
        self.cv = cv
        self.multiclass_strategy = config["classification"]["multiclass_strategy"]

        models_config = config["models"]["logistic_regression"]

        if binaire:
            # Solver binaire
            self.model = LogisticRegression(
                solver=models_config["solvers"]["binary"],
                max_iter=models_config["max_iter"]
            )
            # Grille pour GridSearchCV
            self.param_grid = {
                "C": models_config["C"],
                "penalty": models_config["penalty"],
            }
        else:
            # Multiclasse : base logistic + OVO / OVR wrapper
            base_model = LogisticRegression(
                solver=models_config["solvers"]["multi"],
                max_iter=models_config["max_iter"]
            )

            if self.multiclass_strategy == "ovo":
                self.model = OneVsOneSplit(base_model)
            else:  # 'ovr' par défaut
                self.model = OneVsRestSplit(base_model)

            # Grille adaptée aux meta-estimators
            self.param_grid = {
                "estimator__C": models_config["C"]
            }

        self.best_model = None

    # ------------------------ Prétraitement étiquettes ------------------------
    def pretraiter_etiquettes(self, y):
        """
        Si binaire : map 0 -> 0, le reste -> 1 (comme dans ton code).
        Sinon : labels inchangés.
        """
        if self.binaire:
            y = np.asarray(y)
            return np.array([0 if etiquette == 0 else 1 for etiquette in y])
        else:
            return y

    # ------------------------ Hyperparameter tuning ---------------------------
    def tune_hyperparameters(self, X_train, y_train):
        """
        Effectue un GridSearchCV et met à jour self.best_model.
        """
        print("Starting LogisticRegression hyperparameter tuning...")

        grid = GridSearchCV(
            self.model,
            self.param_grid,
            cv=self.cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        self.best_model = grid.best_estimator_
        print("Best parameters:", grid.best_params_)

    # ------------------------------ Training ----------------------------------
    def train(self, X_train, y_train, tune=True):
        """
        Prétraite les labels, puis :
        - si tune=True : GridSearch + fit
        - sinon : fit direct sur le modèle de base
        """
        # Convertir DataFrame -> ndarray si besoin
        X_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        y_np = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)

        y_transforme = self.pretraiter_etiquettes(y_np)

        if tune:
            self.tune_hyperparameters(X_np, y_transforme)
        else:
            self.best_model = self.model
            self.best_model.fit(X_np, y_transforme)

        train_acc = self.best_model.score(X_np, y_transforme)
        print(f"Training Accuracy: {train_acc:.4f}")

    # ------------------------------ Evaluation --------------------------------
    def evaluate(self, X_test, y_test):
        """
        Calcule accuracy, precision, recall, F1, AUC, matrice de confusion.
        Retourne un dict.
        """
        if self.best_model is None:
            raise RuntimeError("Le modèle doit être entraîné avant l'évaluation.")

        X_np = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
        y_np = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

        y_transforme = self.pretraiter_etiquettes(y_np)
        y_pred = self.best_model.predict(X_np)

        accuracy = accuracy_score(y_transforme, y_pred)
        precision = precision_score(y_transforme, y_pred, average="weighted")
        recall = recall_score(y_transforme, y_pred, average="weighted")
        f1 = f1_score(y_transforme, y_pred, average="weighted")

        # AUC-ROC
        if hasattr(self.best_model, "predict_proba"):
            try:
                y_probs = self.best_model.predict_proba(X_np)
                if y_probs.shape[1] > 2:
                    auc_roc = roc_auc_score(y_transforme, y_probs, multi_class="ovr")
                else:
                    auc_roc = roc_auc_score(y_transforme, y_probs[:, 1])
            except Exception as e:
                print(f"Error during AUC computation: {e}")
                auc_roc = "N/A"
        else:
            auc_roc = "N/A"

        conf_mat = confusion_matrix(y_transforme, y_pred)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc}")
        print(f"Confusion Matrix:\n{conf_mat}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "confusion_matrix": conf_mat,
        }

    # ------------------------------ Prédiction --------------------------------
    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Le modèle n'a pas encore été entraîné.")

        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        return self.best_model.predict(X_np)

    def predict_proba(self, X):
        if self.best_model is None:
            raise RuntimeError("Le modèle n'a pas encore été entraîné.")

        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        return self.best_model.predict_proba(X_np)
