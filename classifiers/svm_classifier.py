from classifiers.base_classifier import BaseClassifier
from sklearn.svm import SVC


class SVMClassifier(BaseClassifier):
    """
    Implémentation du classifieur Support Vector Machine.
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', probability=True):
        """
        Initialise le classifieur SVM.
        
        Args:
            C (float): Paramètre de régularisation
            kernel (str): Type de noyau ('linear', 'poly', 'rbf', 'sigmoid')
            gamma (str ou float): Coefficient du noyau pour 'rbf', 'poly' et 'sigmoid'
            probability (bool): Active le calcul des probabilités
        """
        super().__init__(name="Support Vector Machine")
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
    
    def build_model(self):
        """
        Construit le modèle SVM avec les paramètres actuels.
        """
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=self.probability,
            random_state=42
        )
        return self.model
    
    def get_param_grid(self):
        """
        Définit la grille d'hyperparamètres à optimiser.
        
        Returns:
            dict: Dictionnaire des hyperparamètres pour GridSearchCV
        """
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
        }
        return param_grid 