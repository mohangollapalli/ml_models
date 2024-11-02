
import numpy as np  # type: ignore

def sigmoid(z): 
    return 1/(1+np.exp(-z))

class LogisticRegression: 
    def __init__(self, lr=0.001, n_iters=1000, weights=None, bias=None): 
        self.lr = lr 
        self.n_iters = n_iters 
        self.weights = weights 
        self.bias = bias 
    
    def fit(self, X, y):  # X_train, y_train
        n_samples, n_features = X.shape 

        # Initiaze the weights and bias 
        self.weights = np.zeros(n_features)
        self.bias = 0 

        # Generate preds 
        for iter in range(self.n_iters): 
            z = np.dot(X, self.weights) + self.bias 
            y_preds = sigmoid(z)

            dw = (1/n_samples)*(np.dot(X.T, (y_preds - y)))
            db = (1/n_samples)*(np.sum(y_preds - y))

            self.weights = self.weights - (self.lr*dw)
            self.bias = self.bias - (self.lr*db)
    
    def predict(self, X): # X_test 
        z = np.dot(X, self.weights) + self.bias 
        y_preds = sigmoid(z)
        return y_preds

        