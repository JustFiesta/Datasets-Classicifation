from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from collections import Counter

class ManualKNN:
    def __init__(self, k=3, weights='uniform', metric='euclidean', p=2):
        """
        Initialize KNN classifier with parameters
        
        Args:
            k (int): Number of neighbors
            weights (str): Weight function {'uniform', 'distance'}
            metric (str): Distance metric {'euclidean', 'manhattan', 'minkowski'}
            p (int): Power parameter for Minkowski metric
        """
        self.k = k
        self.weights = weights
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None
    
    def _calculate_weights(self, distances):
        """Calculate weights for neighbors based on their distances"""
        if self.weights == 'uniform':
            return np.ones(distances.shape)
        elif self.weights == 'distance':
            # Add small constant to avoid division by zero
            return 1 / (distances + 1e-10)
        else:
            raise ValueError(f"Unknown weight function: {self.weights}")
    
    def _calculate_distances(self, X1, X2):
        """Calculate distances between points using specified metric"""
        if self.metric == 'euclidean':
            # Euclidean distance using matrix operations
            a2 = np.sum(X1**2, axis=1, keepdims=True)
            b2 = np.sum(X2**2, axis=1)
            ab = np.dot(X1, X2.T)
            distances = np.sqrt(np.maximum(a2 + b2 - 2*ab, 0))
        elif self.metric == 'manhattan':
            # Manhattan distance
            distances = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                distances[i] = np.sum(np.abs(X2 - X1[i]), axis=1)
        elif self.metric == 'minkowski':
            # Minkowski distance
            distances = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                distances[i] = np.power(np.sum(np.power(np.abs(X2 - X1[i]), self.p), axis=1), 1/self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        return distances
    
    def fit(self, X_train, y_train):
        """Store training data as np.array"""
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
    
    def predict_batch(self, X_test, batch_size=100):
        """
        Predict labels for test points using vectorized operations
        
        Args:
            X_test (array-like): Test data
            batch_size (int): Size of batches for memory efficiency
        """
        n_test = X_test.shape[0]
        predictions = np.zeros(n_test)
        
        for i in range(0, n_test, batch_size):
            end_idx = min(i + batch_size, n_test)
            batch = X_test[i:end_idx]
            
            # Calculate distances using specified metric
            distances = self._calculate_distances(batch, self.X_train)
            
            # Find k nearest neighbors
            k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
            k_distances = np.take_along_axis(distances, k_indices, axis=1)
            
            # Calculate weights
            weights = self._calculate_weights(k_distances)
            
            # Weighted voting for each point in batch
            for j, (indices, point_weights) in enumerate(zip(k_indices, weights)):
                labels = self.y_train[indices]
                if self.weights == 'uniform':
                    predictions[i + j] = Counter(labels).most_common(1)[0][0]
                else:
                    # Weighted voting
                    weighted_votes = {}
                    for label, weight in zip(labels, point_weights):
                        weighted_votes[label] = weighted_votes.get(label, 0) + weight
                    predictions[i + j] = max(weighted_votes.items(), key=lambda x: x[1])[0]
        
        return predictions
    
    def predict(self, X):
        """Predict labels for all test points"""
        X = np.asarray(X)
        return self.predict_batch(X)

def manual_knn_classifier(X_train, X_test, y_train, y_test, **kwargs):
    """
    Wrapper function for manual KNN implementation
    
    Args:
        X_train, X_test: Training and test data
        y_train, y_test: Training and test labels
        **kwargs: Additional parameters for KNN classifier
            - n_neighbors (int): Number of neighbors
            - weights (str): Weight function {'uniform', 'distance'}
            - metric (str): Distance metric {'euclidean', 'manhattan', 'minkowski'}
            - p (int): Power parameter for Minkowski metric
    
    Returns:
        dict: Dictionary with classification results
    """
    # Extract parameters from kwargs with defaults
    n_neighbors = kwargs.get('n_neighbors', 5)
    weights = kwargs.get('weights', 'uniform')
    metric = kwargs.get('metric', 'euclidean')
    p = kwargs.get('p', 2)
    
    # Convert data to float32 for better performance
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    
    # Normalize data for better numerical stability
    X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-10)
    X_test = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-10)
    
    # Initialize and train model
    knn = ManualKNN(k=n_neighbors, weights=weights, metric=metric, p=p)
    knn.fit(X_train, y_train)
    
    # Predict labels
    y_pred = knn.predict(X_test)
    
    # Calculate metrics
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "y_pred": y_pred,
    }
    
    return results