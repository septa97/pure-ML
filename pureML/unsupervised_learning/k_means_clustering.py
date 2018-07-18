import numpy as np

class KMeansClustering():
    def __init__(self, k=2, max_iter=500, vectorized=True):
        """Initialize a k-means clustering instance.

        Args:
            k: Number of clusters
            max_iter: Maximum number of iterations (incase the algorithm didn't converge)
            vectorized: Custom parameter to test the speed of vectorized operations vs
                        looped operations
        """
        self.k = k
        self.max_iter = max_iter
        self.vectorized = vectorized

    def _euclidean_distance(self, x, centroid):
        """Computes the Euclidean distance of a data point x and a centroid

        Args:
            x: The data point
            centroid: The centroid
        """
        return np.sqrt(np.sum((x - centroid) ** 2))

    def _initialize_centroids(self, X):
        """Initialize the k centroids arbitrarily.

        Args:
            X: The dataset where the k centroids will be picked
        """
        self.n, self.m = X.shape
        centroids = np.empty((self.k, self.m))

        for i in range(self.k):
            centroids[i] = X[np.random.choice(range(self.n))]

        return centroids

    def _update_centroids(self, X, clusters):
        """Update the centroids by computing the mean of the clusters

        Args:
            X: The dataset
            clusters: The currently associated cluster of the dataset
        """
        centroids = np.empty((self.k, self.m))

        for c in range(self.k):
            centroids[c] = np.mean(X[clusters == c, :], axis=0)

        return centroids

    def _assign_nearest_centroid(self, X, centroids):
        """Assign the nearest centroid to each data point (the distance metric used is the Euclidean distance).

        Args:
            X: The dataset that will have their cluster updated based on the k centroids
            centroids: The k centroids
        """
        clusters = np.empty(self.n)

        for i in range(self.n):
            distances = np.empty(self.k)

            for c in range(self.k):
                distances[c] = self._euclidean_distance(X[i, :], centroids[c, :])

            clusters[i] = np.argmin(distances)

        return clusters

    def cluster(self, X):
        """Cluster the dataset provided
        
        Args:
            X: The dataset
        """
        centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            clusters = self._assign_nearest_centroid(X, centroids)
            new_centroids = self._update_centroids(X, clusters)

            # If the centroids doesn't change, it means that the algorithm converge
            diff = centroids - new_centroids
            if not diff.any():
                break

            centroids = new_centroids

        return clusters, centroids


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # By default, there are 2 features for each data point. We'll only use 2 features so that we can easily visualize the dataset.
    k = 3
    X, y = make_blobs(centers=3) 
    plt.scatter(X[:, 0], X[:, 1], color='black')
    plt.title('Unlabeled dataset')
    plt.show()

    for i, label in zip(range(k), 'abcdefghijklmnopqrstuvwxyz'):
        plt.scatter(X[y == i, 0], X[y == i, 1], label=label, lw=2)
    plt.title('Original Clusters')
    plt.show()

    model = KMeansClustering(k=k)
    clusters, centroids = model.cluster(X)

    for i, label in zip(range(k), 'abcdefghijklmnopqrstuvwxyz'):
        plt.scatter(X[clusters == i, 0], X[clusters == i, 1], label=label, lw=2)
    plt.title('Learned Clusters')
    plt.show()

