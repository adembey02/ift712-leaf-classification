from classifiers.base_classifier import BaseClassifier
from sklearn.neural_network import MLPClassifier


class MLPNetClassifier(BaseClassifier):
    """
    Implémentation du classifieur Neural Network (Multi-Layer Perceptron).
    """

    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, learning_rate='constant', learning_rate_init=0.001,
                 max_iter=1000, batch_size='auto'):
        """
        Initialise le classifieur MLP.

        Args:
            hidden_layer_sizes (tuple): Nombre de neurones dans chaque couche cachée
            activation (str): Fonction d'activation ('identity', 'logistic', 'tanh', 'relu')
            solver (str): Solveur pour l'optimisation des poids ('lbfgs', 'sgd', 'adam')
            alpha (float): Terme de régularisation L2
            learning_rate (str): Taux d'apprentissage ('constant', 'invscaling', 'adaptive')
            learning_rate_init (float): Taux d'apprentissage initial
            max_iter (int): Nombre maximum d'itérations
            batch_size (int ou 'auto'): Taille des mini-lots pour les optimiseurs basés sur des gradients
        """
        super().__init__(name="Multi-Layer Perceptron")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size

    def build_model(self):
        """
        Construit le modèle MLP avec les paramètres actuels.
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
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
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
        return param_grid 