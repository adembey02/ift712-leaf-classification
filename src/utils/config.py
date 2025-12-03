# Configuration settings for the IFT712 classification project

# File paths
DATA_RAW_PATH = 'data/raw/'
DATA_PROCESSED_PATH = 'data/processed/'

# Hyperparameters
HYPERPARAMETERS = {
    'n_estimators': 100,
    'max_depth': None,
    'learning_rate': 0.01,
    'random_state': 42
}

# Classifier settings
CLASSIFIER_SETTINGS = {
    'classifier_type': 'RandomForest',  # Options: 'RandomForest', 'SVC', 'LogisticRegression', etc.
    'use_cross_validation': True,
    'cv_folds': 5
}