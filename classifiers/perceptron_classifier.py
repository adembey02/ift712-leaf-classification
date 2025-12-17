from sklearn.linear_model import Perceptron
from .base_classifier import BaseClassifier


class PerceptronClassifier(BaseClassifier):
    """
    Classifieur Perceptron.
    """
    
    def __init__(self, penalty=None, alpha=0.0001, max_iter=1000, tol=1e-3, shuffle=True, eta0=1.0):
        """
        Initialise le classifieur Perceptron.
        
        Args:
            penalty (str): Terme de pénalité à utiliser (l1, l2, elasticnet ou None)
            alpha (float): Constante qui multiplie le terme de pénalisation
            max_iter (int): Nombre maximum d'itérations
            tol (float): Critère d'arrêt
            shuffle (bool): Si True, le jeu d'apprentissage est mélangé à chaque itération
            eta0 (float): Taux d'apprentissage constant
        """
        super().__init__(name="Perceptron")
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.eta0 = eta0
        self.model = None
    
    def build_model(self):
        """
        Construit le modèle Perceptron.
        
        Returns:
            Perceptron: Le modèle Perceptron configuré
        """
        self.model = Perceptron(
            penalty=self.penalty,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=self.shuffle,
            eta0=self.eta0,
            random_state=42
        )
        return self.model
    
    def get_param_grid(self):
        """
        Définit la grille de paramètres pour l'optimisation des hyperparamètres.
        
        Returns:
            dict: Grille de paramètres pour GridSearchCV
        """
        param_grid = {
            'penalty': [None, 'l1', 'l2', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [500, 1000, 2000],
            'eta0': [0.1, 1.0, 2.0]
        }
        
        return param_grid 