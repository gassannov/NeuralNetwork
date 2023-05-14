import numpy as np


# from ../dz1/dz import Linear, Sigmoid, NLLLoss, NeuralNetwork # Results from Seminar 1
class Linear:
    def __init__(self, input_size, output_size):
        '''
        Creates weights and biases for linear layer.
        Dimention of inputs is *input_size*, of output: *output_size*.
        '''
        self.W = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros(output_size)
        pass

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size).
        Returns output of size (N, output_size).
        Hint: You may need to store X for backward pass
        '''
        self.X = X
        self.Y = X @ self.W + self.b
        return self.Y

    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        self.dLdX = dLdy @ self.W.T
        self.dLdw = np.expand_dims(self.X, -1) * np.expand_dims(dLdy, -2)
        self.dLdb = dLdy

        if len(self.dLdw.shape) > 2:
            self.dLdw = self.dLdw.sum(0)
        if len(self.dLdb.shape) > 1:
            self.dLdb = self.dLdb.sum(0)

        return self.dLdX

    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - l*dLdw
        '''
        self.W = self.W - learning_rate*self.dLdw
        self.b = self.b - learning_rate*self.dLdb


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        self.X = X
        self.Y = 1/(1+np.exp(-X))
        return self.Y

    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        self.dLdx = dLdy * (self.Y*(1- self.Y))
        return self.dLdx

    def step(self, learning_rate):
        pass


class NLLLoss:
    def init(self):
        '''
        Applies Softmax operation to inputs and computes NLL loss
        '''
        print("some")
        #### YOUR CODE HERE
        #### (Hint: No code is expected here, just joking)
        pass

    def forward(self, X, y):
        '''
        Passes objects through this layer.
        X is np.array of size (N, C), where C is the number of classes
        y is np.array of size (N), contains correct labels
        '''
        self.L = -X[range(np.shape(X)[0]), y] + np.log(np.sum(np.exp(X), axis=1))
        self.X = X
        self.y = y
        return self.L

    def get_some(self):
        return 5

    def backward(self):
        '''
        Note that here dLdy = 1 since L = y
        1. Compute dLdx
        2. Return dLdx
        '''
        self.dLdx = np.exp(self.X)/np.tile( np.sum(np.exp(self.X), axis=1).reshape(np.shape(self.X)[0],1), (1, np.shape(self.X)[1]))
        self.dLdx[range(np.shape(self.X)[0]), self.y] -= 1
        return self.dLdx


class NeuralNetwork:
    def __init__(self, modules):
        """
        Constructs network with *modules* as its layers
        """
        # YOUR CODE HERE
        self.layers = modules
        pass

    def forward(self, X):
        # YOUR CODE HERE
        # Apply layers to input
        currX = X
        for layer in self.layers:
            currX = layer.forward(currX)
        return currX

    def backward(self, dLdy):
        """
        dLdy here is a gradient from loss function
        """
        # YOUR CODE HERE
        currdLdy = dLdy
        for layer in reversed(self.layers):
            currdLdy = layer.backward(currdLdy)

    def step(self, learning_rate):
        for layer in self.layers:
            layer.step(learning_rate)

    def predict(self, X):
        pred = self.forward(X)
        return np.argmax(pred, axis=1)


# %%
