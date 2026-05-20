import numpy as np


class LogisticRegressionScratch:
    """
    Logistic regression classifier implemented from scratch using NumPy.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold

        self.w = None
        self.b = None
        self.losses = []

    def sigmoid(self, z):
        """
        Apply the sigmoid function to map raw scores to probabilities.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_linear_model(self, X):
        """
        Compute the linear model output z = Xw + b.
        """
        return np.dot(X, self.w) + self.b

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.
        """
        linear_model = self.compute_linear_model(X)

        return self.sigmoid(linear_model)

    def predict(self, X):
        """
        Predict binary class labels using the configured probability threshold.
        """
        y_proba = self.predict_proba(X)

        return (y_proba >= self.threshold).astype(int)

    def compute_loss(self, y, y_pred):
        """
        Compute binary cross-entropy loss.
        """
        n_samples = len(y)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -(1 / n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        return loss

    def compute_gradients(self, X, y, y_pred):
        """
        Compute gradients of the loss with respect to weights and bias.
        """
        n_samples = len(y)
        error = y_pred - y

        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)

        return dw, db

    def update_weights(self, dw, db):
        """
        Update model parameters using gradient descent.
        """
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def fit(self, X, y):
        """
        Train logistic regression using gradient descent.
        """
        _, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.losses = []

        for _ in range(self.n_iterations):
            y_pred = self.predict_proba(X)
            loss = self.compute_loss(y, y_pred)
            dw, db = self.compute_gradients(X, y, y_pred)

            self.update_weights(dw, db)

            self.losses.append(loss)

        return self
