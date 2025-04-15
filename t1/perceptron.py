import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the perceptron model.

        Parameters:
            X (ndarray): Training data of shape (n_samples, n_features).
            y (ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert y to {-1, 1} if necessary
        y = np.where(y <= 0, -1, 1)
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)
                
                # Update rule
                if y_predicted != y[idx]:
                    self.weights += self.learning_rate * y[idx] * x_i
                    self.bias += self.learning_rate * y[idx]

    def predict(self, X):
        """
        Predict the class labels for input data.

        Parameters:
            X (ndarray): Input data of shape (n_samples, n_features).

        Returns:
            ndarray: Predicted class labels.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

# Example usage
if __name__ == "__main__":
    # Sample data (AND gate example)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    # Initialize and train perceptron
    perceptron = Perceptron(learning_rate=0.1, n_iterations=10)
    perceptron.fit(X, y)

    # Make predictions
    predictions = perceptron.predict(X)
    print("Predictions:", predictions)