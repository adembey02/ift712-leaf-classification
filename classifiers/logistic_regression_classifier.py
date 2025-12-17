from classifiers.base_classifier import BaseClassifier
from sklearn.linear_model import LogisticRegression


class LRClassifier(BaseClassifier):
    """
    Implémentation du classifieur Logistic Regression.
    """
    
    def __init__(self, C=1.0, penalty='l2', solver='lbfgs', max_iter=5000):
        """
        Initialise le classifieur Logistic Regression.
        
        Args:
            C (float): Inverse de la force de régularisation
            penalty (str): Type de pénalité ('l1', 'l2', 'elasticnet', 'none')
            solver (str): Algorithme d'optimisation ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
            max_iter (int): Nombre maximum d'itérations (augmenté à 5000 pour éviter les problèmes de convergence)
        """
        super().__init__(name="Logistic Regression")
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
    
    def build_model(self):
        """
        Construit le modèle Logistic Regression avec les paramètres actuels.
        """
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=42,
            n_jobs=-1
        )
        return self.model
    
    def get_param_grid(self):
        """
        Définit la grille d'hyperparamètres à optimiser.
        
        Returns:
            dict: Dictionnaire des hyperparamètres pour GridSearchCV
        """
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],  # 'l1' seulement avec solver='liblinear' ou 'saga'
            'solver': ['lbfgs', 'newton-cg', 'saga'],
            'max_iter': [2000, 5000]  # Valeurs plus élevées pour éviter les problèmes de convergence
        }
        return param_grid 