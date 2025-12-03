# Main entry point of the application for the IFT712 classification project

import os
from src.data.loader import load_data
from src.preprocessing.preprocessor import DataPreprocessor
from src.classifiers.classifier_base import ClassifierBase
from src.evaluation.evaluator import evaluate_model
from src.utils.config import Config

def main():
    # Load configuration settings
    config = Config()

    # Load raw data
    raw_data = load_data(config.raw_data_path)

    # Preprocess the data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process(raw_data)

    # Initialize classifiers
    classifiers = [ClassifierBase()]

    # Train and evaluate each classifier
    for classifier in classifiers:
        classifier.train(processed_data)
        results = evaluate_model(classifier, processed_data)
        print(f"Results for {classifier.__class__.__name__}: {results}")

if __name__ == "__main__":
    main()