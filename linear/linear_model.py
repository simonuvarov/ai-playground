import logging

import numpy as np


class LinearRegression():
    def __init__(self):
      self.n_features = None
      self.n_samples = None
      self.learning_data = []
      self.theta = []

    def _add_intercept_column(self, arr):
        m, _ = np.shape(arr)
        return np.c_[np.ones(m, dtype=int), arr]

    def init(self, X, y):
        m, n = np.shape(X)
        self.n_samples = m
        self.n_features = n
        self.learning_data = []
        self.X = np.c_[np.ones(m, dtype=int), X]
        self.y = np.reshape(y, (m, 1))
        self.theta = np.ones((n+1,1))

    def calculate_gradient(self):
        # y = np.reshape(y, (m, 1))
        predicted = self.X.dot(self.theta)
        result = 2/self.n_samples * self.X.T.dot(predicted - self.y)
        return result

    def calculate_loss(self):
        return np.square(self.X.dot(self.theta) - self.y)

    def calculate_cost(self, loss):
        return np.sum(loss)/self.n_samples

    def fit(self, X, y, max_steps=10000, learning_rate=0.001):
        self.init(X, y)
        for epoch in range(0, max_steps):
            loss = self.calculate_loss()
            cost = self.calculate_cost(loss)
            self.learning_data.append((self.theta,cost))
            gradient = self.calculate_gradient()
            logging.info(f'Epoch {epoch+1}: {cost}')
            self.theta = self.theta - learning_rate * gradient

    def predict(self, X):
        X = self._add_intercept_column(X)
        result = X @ self.theta
        return result
