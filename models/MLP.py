import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


class TorchMLP(BaseEstimator, ClassifierMixin):
    """
    Implémentation d'un MLP PyTorch compatible scikit-learn.
    """

    def __init__(
        self,
        hidden_sizes=(256, 128),
        dropout_rate=0.5,
        learning_rate=0.001,
        epochs=20,
        batch_size=64,
        device=None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        self.model = None
        self.classes_ = None

    def _build_model(self, input_size, output_size):
        layers = []
        prev_size = input_size
        for size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y):
        """
        Entraîne le MLP sur (X, y).
        Compatible avec GridSearchCV.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.classes_ = np.unique(y)
        input_size = X.shape[1]
        output_size = len(self.classes_)

        self.model = self._build_model(input_size, output_size)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        """
        Prédit les classes.
        """
        self.model.eval()
        X = np.asarray(X)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()

    def predict_proba(self, X):
        """
        Probabilités par classe.
        """
        self.model.eval()
        X = np.asarray(X)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()


class MLPModel:
    """
    Enveloppe pour TorchMLP avec :
    - grille d'hyperparamètres (GridSearchCV)
    - train()
    - evaluate()
    """

    def __init__(self, param_grid=None, cv=5):
        # modèle de base
        self.model = TorchMLP()
        self.cv = cv

        # grille par défaut si rien fourni
        if param_grid is None:
            param_grid = {
                "hidden_sizes": [(256, 128), (512, 256), (128,)],
                "dropout_rate": [0.3, 0.5],
                "learning_rate": [0.001, 0.0001],
                "batch_size": [64, 128],
                "epochs": [20, 30],
            }

        self.param_grid = param_grid
        self.best_model = None

    # ---------------- HYPERPARAM TUNING ----------------
    def tune_hyperparameters(self, X_train, y_train):
        """
        GridSearchCV sur TorchMLP.
        """
        print("Starting MLP hyperparameter tuning (GridSearchCV)...")

        grid = GridSearchCV(
            self.model,
            self.param_grid,
            cv=self.cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        self.best_model = grid.best_estimator_
        print("Best parameters found:", grid.best_params_)

    # ---------------- TRAIN ----------------
    def train(self, X_train, y_train, tune=True):
        """
        Entraîne le MLP (avec ou sans tuning).
        """
        X_np = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        y_np = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)

        if tune:
            self.tune_hyperparameters(X_np, y_np)
        else:
            self.best_model = self.model
            self.best_model.fit(X_np, y_np)

        train_accuracy = self.best_model.score(X_np, y_np)
        print(f"Training Accuracy: {train_accuracy:.4f}")

    # ---------------- EVALUATE ----------------
    def evaluate(self, X_test, y_test):
        """
        Même style que BaseModel.evaluate :
        accuracy, precision, recall, f1, AUC, confusion matrix.
        """
        if self.best_model is None:
            raise RuntimeError("Model must be trained before evaluation.")

        X_np = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
        y_np = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

        y_pred = self.best_model.predict(X_np)

        accuracy = accuracy_score(y_np, y_pred)
        precision = precision_score(y_np, y_pred, average="weighted")
        recall = recall_score(y_np, y_pred, average="weighted")
        f1 = f1_score(y_np, y_pred, average="weighted")

        # AUC-ROC si proba dispo
        if hasattr(self.best_model, "predict_proba"):
            try:
                y_probs = self.best_model.predict_proba(X_np)
                if y_probs.shape[1] > 2:
                    auc_roc = roc_auc_score(y_np, y_probs, multi_class="ovr")
                else:
                    auc_roc = roc_auc_score(y_np, y_probs[:, 1])
            except Exception as e:
                print(f"Error during AUC computation: {e}")
                auc_roc = "N/A (error during predict_proba)"
        else:
            auc_roc = "N/A (model does not support probability predictions)"

        conf_matrix = confusion_matrix(y_np, y_pred)

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

    # ---------------- PREDICT WRAPPERS ----------------
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
