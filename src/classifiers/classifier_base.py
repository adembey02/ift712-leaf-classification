class ClassifierBase:
    def __init__(self):
        """Initialize the base classifier."""
        self.model = None

    def train(self, X, y):
        """
        Train the classifier on the provided data.

        Parameters:
        X : array-like
            Feature data for training.
        y : array-like
            Target labels for training.
        """
        raise NotImplementedError("Train method must be implemented by subclasses.")

    def predict(self, X):
        """
        Predict the labels for the provided data.

        Parameters:
        X : array-like
            Feature data for prediction.

        Returns:
        array-like
            Predicted labels.
        """
        raise NotImplementedError("Predict method must be implemented by subclasses.")

    def score(self, X, y):
        """
        Evaluate the classifier on the provided data.

        Parameters:
        X : array-like
            Feature data for evaluation.
        y : array-like
            True labels for evaluation.

        Returns:
        float
            Accuracy of the classifier.
        """
        predictions = self.predict(X)
        return (predictions == y).mean()  # Simple accuracy calculation