import numpy as np

class PseudoInverse:
    def fit(self, X, y):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        self.w = np.linalg.pinv(X_aug) @ y
        return self
    
    def predict(self, X):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.sign(X_aug @ self.w)
    
    def decision_function(self, X):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return X_aug @ self.w


class GradientDescent:
    def __init__(self, lr=0.01, epochs=1000, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        self.w = np.zeros(n_features + 1)
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                X_batch = X_aug[batch_idx]
                y_batch = y[batch_idx]
                
                y_pred = X_batch @ self.w
                grad = 2 * X_batch.T @ (y_pred - y_batch) / len(batch_idx)
                self.w -= self.lr * grad
                
                loss = np.mean((y_pred - y_batch) ** 2)
                epoch_loss += loss
            
            self.losses.append(epoch_loss / (n_samples // self.batch_size + 1))
        
        return self
    
    def predict(self, X):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.sign(X_aug @ self.w)
    
    def decision_function(self, X):
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return X_aug @ self.w
