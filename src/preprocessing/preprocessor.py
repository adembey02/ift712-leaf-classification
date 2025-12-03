class Preprocessor:
    def __init__(self):
        pass

    def normalize(self, data):
        """Normalize the data."""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def encode(self, data, categorical_features):
        """Encode categorical features."""
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse=False)
        encoded_features = encoder.fit_transform(data[categorical_features])
        return encoded_features

    def split_data(self, data, target, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        from sklearn.model_selection import train_test_split
        return train_test_split(data, target, test_size=test_size, random_state=random_state)