import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def safe_log(x):
    return np.log(x.clip(min=0.0000000001))

class LogisticRegression():
    def __init__(self, batch_size=None, learning_rate=0.001, num_epoch=4000, threshold=0.5, vectorized=True):
        """Initialize a logistic regression instance.

        Args:
            batch_size: Number of rows per batch (step) that will be used in gradient computation
                        (for SGD, batch_size=1 while for mini-batch SGD, batch_size=[10,1000])
            learning_rate: A hyperparameter which is used to multiply to the gradient
            num_epoch: Number of pass on the whole dataset
            threshold: The middle value between the two classes after the execution of the sigmoid function
            vectorized: Custom parameter to test the speed of vectorized operations vs
                        looped operations
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.threshold = threshold
        self.vectorized = vectorized

    def loss_function(self, y, y_pred):
        return np.mean(y.dot(safe_log(y_pred)) + (1 - y).dot(safe_log(1 - y_pred)))

    def train(self, X, y, shuffle=False):
        """Trains a logistic regression model.

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
                y_pred = sigmoid(np.matmul(X, self.weights))

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
            h = sigmoid(np.matmul(X, self.weights))
            p = [1 if h[i] >= self.threshold else 0 for i in range(X.shape[0])]

            return np.array(p)


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    import matplotlib.pyplot as plt

    X = load_breast_cancer().data
    y = load_breast_cancer().target

    data_size = y.shape[0]
    split = int(data_size * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LogisticRegression()
    model.train(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # TODO: Put this function on a separate module
    def accuracy_score(y, y_pred):
        return sum(y == y_pred) / y.shape[0]

    training_accuracy = accuracy_score(y_train, y_train_pred) * 100
    testing_accuracy = accuracy_score(y_test, y_test_pred) * 100

    # TODO: Visualize the dataset using PCA

    plt.plot(range(len(model.training_errors)), model.training_errors, color='black', linewidth='3')
    plt.title('Final Training Error: %.2f\nTraining accuracy: %.2f\nTesting accuracy: %.2f' % (model.training_errors[-1], training_accuracy, testing_accuracy))
    plt.xlabel('Epoch')
    plt.ylabel('Training Error')
    plt.show()

