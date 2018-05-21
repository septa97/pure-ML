import numpy as np

class MultivariateRegression():
    def __init__(self, batch_size=None, learning_rate=0.001, loss='l2', num_epoch=100, vectorized=True):
        """Initialize a multivariate regression instance.

        Args:
            batch_size: Number of rows per batch (step) that will be used in gradient computation
                        (for SGD, batch_size=1 while for mini-batch SGD, batch_size=[10,1000])
            learning_rate: A hyperparameter which is used to multiply to the gradient
            loss: Loss function to be used in computing the error
            num_epoch: Number of pass on the whole dataset
            vectorized: Custom parameter to test the speed of vectorized operations vs
                        looped operations
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.vectorized = vectorized

        if loss == 'l1':
            self.loss_function = lambda y, y_pred: np.mean(abs(y - y_pred))
        elif loss == 'l2':
            self.loss_function = lambda y, y_pred: np.mean((y - y_pred) ** 2)

    def train(self, X, y, shuffle=False):
        """Trains a multivariate regression model.

        Args:
            X: The nxm matrix where n is the number of rows
                and m is the number of features (including the bias weight)
            y: A vector of size n which corresponds to the label of each row in the X matrix
            shuffle: Shuffle the dataset before training or not
        """

        # TODO: Add shuffling of X and y

        # Add the bias column
        X = np.insert(X, 0, 1, axis=1)
        self.n, self.m = X.shape
        self.training_errors = []
        self.weights = np.zeros(self.m)

        # TODO: Add a choice for all zero weights or random uniformly distributed weights
        # limit = 1 / math.sqrt(self.m)
        # self.weights = np.random.uniform(-limit, limit, (self.m, ))

        # TODO: Feature scaling, must be in a separate module

        if self.batch_size is None:
            self.batch_size = X.shape[0]

        if self.vectorized:
            for i in range(self.num_epoch):
                y_pred = X.dot(self.weights)

                error = self.loss_function(y, y_pred)
                self.training_errors.append(error)

                # Compute gradient
                grad = -(y - y_pred).dot(X)

                # Update weights
                self.weights -= self.learning_rate * grad

    def predict(self, X):
        if self.vectorized:
            # Add the bias column
            X = np.insert(X, 0, 1, axis=1)

            return X.dot(self.weights)


if __name__ == "__main__":
    from sklearn.datasets import load_boston, load_diabetes
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt

    X = load_diabetes().data[:, np.newaxis, 2]
    y = load_diabetes().target

    data_size = y.shape[0]
    split = int(data_size * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = MultivariateRegression(loss='l2', num_epoch=5000)
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)

    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.show()

    plt.plot(range(len(model.training_errors)), model.training_errors, color='black', linewidth='3')
    plt.title('Final Training Error: %.2f' % model.training_errors[-1])
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')

    plt.show()

