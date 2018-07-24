import numpy as np

class PCA():
    def __init__(self, k=2, vectorized=True):
        """Initialize a Principal Components Analysis (PCA) instance.

        Args:
            k: Resulting number of dimensions
            vectorized: Custom parameter to test the speed of vectorized operations vs
                        looped operations
        """
        self.k = k
        self.vectorized = vectorized

    def _normalize_and_scale(self, X):
        """Perform mean normalization and feature scaling on the dataset

        Args:
            X: The dataset
        """
        for j in range(self.m):
            X[:, j] = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])

        return X

    def reduce_dimension(self, X):
        """Reduce the dimension of the given dataset.

        Args:
            X: The dataset
        """
        self.n, self.m = X.shape

        X = self._normalize_and_scale(X)
        sigma = 1/self.n * np.matmul(X.T, X)
        u, s, vh = np.linalg.svd(sigma)

        u_reduced = u[:, :self.k]
        Z = np.matmul(X, u_reduced)

        return Z


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt

    iris = load_iris()
    X = iris.data
    y = iris.target

    k = 2
    model = PCA(k=k)
    X_reduced = model.reduce_dimension(X)
    
    for i, label in zip([0, 1, 2], iris.target_names):
        plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], label=label, lw=2)
    plt.legend()
    plt.title('Dimension-reduced dataset')
    plt.show()
