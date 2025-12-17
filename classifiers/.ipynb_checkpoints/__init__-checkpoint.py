from classifiers.knn_classifier import KNNClassifier
from classifiers.random_forest_classifier import RFClassifier
from classifiers.svm_classifier import SVMClassifier
from classifiers.logistic_regression_classifier import LRClassifier
from classifiers.mlp_classifier import MLPNetClassifier
from classifiers.perceptron_classifier import PerceptronClassifier

# Liste de tous les classifieurs disponibles
CLASSIFIERS = [
    KNNClassifier,
    SVMClassifier,
    LRClassifier,      # Logistic Regression
    PerceptronClassifier,
    MLPNetClassifier,
    RFClassifier
] 