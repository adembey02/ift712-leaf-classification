import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle


class BaseClassifier(ABC):
    """
    Classe abstraite de base pour tous les classifieurs.
    """
    
    def __init__(self, name):
        """
        Initialise un classifieur avec un nom.
        
        Args:
            name (str): Nom du classifieur
        """
        self.name = name
        self.model = None
        self.best_params = None
        self.is_fitted = False
    
    @abstractmethod
    def build_model(self):
        """
        Méthode abstraite pour construire le modèle spécifique.
        Doit être implémentée par les sous-classes.
        """
        pass
    
    def train(self, X_train, y_train):
        """
        Entraîne le modèle sur les données fournies.
        
        Args:
            X_train (array-like): Caractéristiques d'entraînement
            y_train (array-like): Étiquettes d'entraînement
        """
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"Le modèle {self.name} a été entraîné avec succès.")
    
    def predict(self, X):
        """
        Prédit les étiquettes pour de nouvelles données.
        
        Args:
            X (array-like): Caractéristiques à prédire
        
        Returns:
            array: Prédictions d'étiquettes
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant de faire des prédictions.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur les données de test.
        
        Args:
            X_test (array-like): Caractéristiques de test
            y_test (array-like): Étiquettes de test
        
        Returns:
            dict: Dictionnaire contenant différentes métriques d'évaluation
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant l'évaluation.")
        
        y_pred = self.predict(X_test)
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\nÉvaluation du modèle {self.name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Création du rapport de classification
        print("\nRapport de classification détaillé:")
        print(classification_report(y_test, y_pred))
        
        return {
            'name': self.name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred
        }
    
    def cross_validate(self, X, y, cv=5):
        """
        Effectue une validation croisée sur le modèle.
        
        Args:
            X (array-like): Caractéristiques
            y (array-like): Étiquettes
            cv (int): Nombre de plis pour la validation croisée
        
        Returns:
            dict: Résultats de la validation croisée
        """
        if self.model is None:
            self.build_model()
        
        # Réalisation de la validation croisée
        accuracy_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        precision_scores = cross_val_score(self.model, X, y, cv=cv, scoring='precision_weighted')
        recall_scores = cross_val_score(self.model, X, y, cv=cv, scoring='recall_weighted')
        f1_scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_weighted')
        
        # Calcul des moyennes et écarts-types
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        mean_precision = np.mean(precision_scores)
        std_precision = np.std(precision_scores)
        mean_recall = np.mean(recall_scores)
        std_recall = np.std(recall_scores)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        print(f"\nRésultats de la validation croisée {cv}-fold pour {self.name}:")
        print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Precision: {mean_precision:.4f} ± {std_precision:.4f}")
        print(f"Recall: {mean_recall:.4f} ± {std_recall:.4f}")
        print(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
        
        return {
            'name': self.name,
            'accuracy': mean_accuracy,
            'accuracy_std': std_accuracy,
            'precision': mean_precision,
            'precision_std': std_precision,
            'recall': mean_recall,
            'recall_std': std_recall,
            'f1': mean_f1,
            'f1_std': std_f1
        }
    
    @abstractmethod
    def get_param_grid(self):
        """
        Méthode abstraite pour définir la grille d'hyperparamètres à optimiser.
        Doit être implémentée par les sous-classes.
        
        Returns:
            dict: Dictionnaire des hyperparamètres pour GridSearchCV
        """
        pass
    
    def optimize_hyperparameters(self, X, y, cv=5):
        """
        Optimise les hyperparamètres du modèle en utilisant GridSearchCV.
        
        Args:
            X (array-like): Caractéristiques
            y (array-like): Étiquettes
            cv (int): Nombre de plis pour la validation croisée
        
        Returns:
            dict: Meilleurs hyperparamètres trouvés
        """
        if self.model is None:
            self.build_model()
        
        param_grid = self.get_param_grid()
        
        print(f"\nRecherche des meilleurs hyperparamètres pour {self.name}...")
        grid_search = GridSearchCV(self.model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        print(f"Meilleurs hyperparamètres trouvés: {self.best_params}")
        print(f"Meilleur score de validation: {grid_search.best_score_:.4f}")
        
        # Mise à jour du modèle avec les meilleurs paramètres
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return self.best_params
    
    def plot_confusion_matrix(self, X_test, y_test, figsize=(10, 8), save_path=None):
        """
        Affiche la matrice de confusion pour le modèle.
        
        Args:
            X_test (array-like): Caractéristiques de test
            y_test (array-like): Étiquettes de test
            figsize (tuple): Taille de la figure
            save_path (str, optional): Chemin pour sauvegarder la figure
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant l'évaluation.")
        
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_test), 
                   yticklabels=np.unique(y_test))
        plt.title(f"Matrice de confusion - {self.name}")
        plt.ylabel('Valeur réelle')
        plt.xlabel('Valeur prédite')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Matrice de confusion enregistrée dans {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_roc_curve(self, X_test, y_test, figsize=(10, 8), save_path=None):
        """
        Affiche la courbe ROC pour le modèle (uniquement pour les problèmes multi-classes).
        
        Args:
            X_test (array-like): Caractéristiques de test
            y_test (array-like): Étiquettes de test
            figsize (tuple): Taille de la figure
            save_path (str, optional): Chemin pour sauvegarder la figure
        """
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant l'évaluation.")
        
        # Convertir y_test en format one-hot encoding
        classes = np.unique(y_test)
        n_classes = len(classes)
        
        # Vérifier si le modèle peut retourner des probabilités
        if not hasattr(self.model, "predict_proba"):
            print(f"Le modèle {self.name} ne prend pas en charge le calcul des probabilités, impossible de tracer la courbe ROC.")
            return
        
        # Binariser les étiquettes
        y_test_bin = label_binarize(y_test, classes=classes)
        
        # Calculer les scores ROC
        y_score = self.model.predict_proba(X_test)
        
        # Calculer la courbe ROC et l'AUC pour chaque classe
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Tracer la courbe ROC pour chaque classe
        plt.figure(figsize=figsize)
        
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink'])
        for i, color, cls in zip(range(n_classes), colors, classes):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'Classe {cls} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title(f'Courbe ROC multi-classe - {self.name}')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
            print(f"Courbe ROC enregistrée dans {save_path}")
        
        plt.show()
        plt.close() 