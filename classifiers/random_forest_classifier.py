from classifiers.base_classifier import BaseClassifier
from sklearn.ensemble import RandomForestClassifier


class RFClassifier(BaseClassifier):
    """
    Implémentation du classifieur Random Forest.
    """
    
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Initialise le classifieur Random Forest.
        
        Args:
            n_estimators (int): Nombre d'arbres dans la forêt
            criterion (str): Fonction pour mesurer la qualité d'une division ('gini' ou 'entropy')
            max_depth (int, optional): Profondeur maximale de l'arbre
            min_samples_split (int): Nombre minimum d'échantillons requis pour diviser un nœud
            min_samples_leaf (int): Nombre minimum d'échantillons requis pour être une feuille
        """
        super().__init__(name="Random Forest")
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
    
    def build_model(self):
        """
        Construit le modèle Random Forest avec les paramètres actuels.
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
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
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        return param_grid 