import numpy as np

class LinearRegression: 
    def __init__(self, lr=0.001, n_iters=1000, weights=None, bias=None): 
        self.lr = lr 
        self.n_iters = n_iters
        self.weights = weights 
        self.bias = bias 
    
    def fit(self, X, y): # X_train, y_train 
        n_samples, n_features = X.shape
        # Initialize the weights and bias parameters 
        self.weights = np.zeros(n_features)
        self.bias = 0 

        for iter in range(self.n_iters): 
            y_preds = np.dot(X, self.weights)+self.bias

            dw = (1/n_samples) * (2*np.dot(X.T, (y_preds - y)))
            db = (1/n_samples) * (2*np.sum(y_preds - y))

            # update the parameters 
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

    def predict(self, X):  # X_test 
        y_preds = np.dot(X, self.weights)+self.bias
        return y_preds