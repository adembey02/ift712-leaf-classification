from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import numpy as np


class KNeighborsModel:
    def __init__(self, config, cv=5):
        """
        Modèle KNN auto-contenu avec tuning, entraînement et évaluation.

        Paramètres
        ----------
        config : dict
            Dictionnaire de configuration (comme avant).
        cv : int
            Nombre de folds pour GridSearchCV.
        """
        models_config = config["models"]["kneighbors_classifier"]

        # Modèle de base avec les 1ères valeurs de la config
        self.model = KNeighborsClassifier(
            n_neighbors=models_config["n_neighbors"][0],
            weights=models_config["weights"][0],
            algorithm=models_config["algorithm"][0],
            leaf_size=models_config["leaf_size"][0],
        )

        # Grille d'hyperparamètres pour GridSearchCV
        self.param_grid = {
            "n_neighbors": models_config["n_neighbors"],
            "weights": models_config["weights"],
            "algorithm": models_config["algorithm"],
            "leaf_size": models_config["leaf_size"],
        }

        self.cv = cv
        self.best_model = None

    # ------------------ OPTIONNEL : prétraitement étiquettes ------------------
    def pretraiter_etiquettes(self, y):
        """
        Ici inchangé pour multiclasse, mais gardé si tu veux
        unifier avec les autres modèles.
        """
        return y

    # ------------------ TUNING ------------------
    def tune_hyperparameters(self, X_train, y_train):
        """
        Lance GridSearchCV pour trouver les meilleurs hyperparamètres.
        """
        print("Starting KNN hyperparameter tuning...")

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

    # ------------------ TRAIN ------------------
    def train(self, X_train, y_train, tune=True):
        """
        Entraîne le modèle KNN.

        tune=True -> fait GridSearchCV avant d'entraîner.
        """
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

    # ------------------ EVALUATE ------------------
    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur X_test, y_test.
        Retourne un dict avec accuracy, precision, recall, f1, AUC, matrice de confusion.
        """
        if self.best_model is None:
            raise RuntimeError("Model must be trained before evaluation.")

        X_np = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
        y_np = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

        y_transforme = self.pretraiter_etiquettes(y_np)
        y_pred = self.best_model.predict(X_np)

        accuracy = accuracy_score(y_transforme, y_pred)
        precision = precision_score(y_transforme, y_pred, average="weighted")
        recall = recall_score(y_transforme, y_pred, average="weighted")
        f1 = f1_score(y_transforme, y_pred, average="weighted")

        # AUC-ROC si proba dispo
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

    # ------------------ PREDICT ------------------
    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError("Model must be trained before predictions.")
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        return self.best_model.predict(X_np)

    def predict_proba(self, X):
        if self.best_model is None:
            raise RuntimeError("Model must be trained before predictions.")
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        return self.best_model.predict_proba(X_np)
