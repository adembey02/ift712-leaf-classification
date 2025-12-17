from classifiers.base_classifier import BaseClassifier
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier(BaseClassifier):
    """
    Implémentation du classifieur K-Nearest Neighbors.
    """

    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        """
        Initialise le classifieur KNN.

        Args:
            n_neighbors (int): Nombre de voisins
            weights (str): Fonction de pondération ('uniform' ou 'distance')
            p (int): Paramètre de puissance pour la distance de Minkowski (1: distance de Manhattan, 2: distance euclidienne)
        """
        super().__init__(name="K-Nearest Neighbors")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def build_model(self):
        """
        Construit le modèle KNN avec les paramètres actuels.
        """
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, p=self.p)
        return self.model

    def get_param_grid(self):
        """
        Définit la grille d'hyperparamètres à optimiser.

        Returns:
            dict: Dictionnaire des hyperparamètres pour GridSearchCV
        """
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        return param_grid 