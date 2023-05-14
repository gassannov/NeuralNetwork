import numpy as np


# from ../dz1/dz import Linear, Sigmoid, NLLLoss, NeuralNetwork # Results from Seminar 1
class Linear:
    def __init__(self, input_size, output_size):
        '''
        Creates weights and biases for linear layer.
        Dimention of inputs is *input_size*, of output: *output_size*.
        '''
        #### YOUR CODE HERE
        #### Create weights, initialize them with samples from N(0, 0.1).
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

        # print("W dim " + str(self.W.shape))
        self.X = X
        # print("X dim " + str(self.X.shape))
        # print("b dim " + str(self.b.shape))
        self.Y = X @ self.W + self.b
        #### YOUR CODE HERE
        #### Apply layer to input
        return self.Y

    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        #### YOUR CODE HERE
        # print("X dim " + str(self.X.shape))
        # print("W dim " + str(self.W.shape))
        # print("b dim " + str(self.b.shape))
        # print("dLdy dim " + str(dLdy.shape))

        self.dLdx = dLdy @ self.W.T
        self.dLdw = np.expand_dims(self.X, -1) * np.expand_dims(dLdy, -2)
        self.dLdw = dLdy.T @ self.X
        self.dLdb = dLdy
        # print("dLx dim " + str(self.dLdx.shape))
        # print("dldW dim " + str(self.dLdw.shape))
        # print("dldb dim " + str(self.dLdb.shape))

        if len(self.dLdb.shape) > 1:
            self.dLdb = self.dLdb.sum(0)
        return self.dLdx

    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - l*dLdw
        '''
        # print(self.W.shape)
        # print(self.b.shape)
        # print(self.dLdw.shape)
        # print(self.dLdb.shape)
        self.W = self.W - learning_rate * self.dLdw.T
        self.b = self.b - learning_rate * self.dLdb


class Sigmoid:
    def __init__(self):
        self.sigmoid_func = lambda x: 1 / (1 + np.exp(-x))

    def forward(self, X):
        """
        Passes objects through this layer.
        X is np.array of size (N, d)
        """
        # YOUR CODE HERE
        self.y = self.sigmoid_func(X)
        self.X = X
        return self.y

    def backward(self, dLdy):
        """
        1. Compute dLdx.
        2. Return dLdx
        """
        # YOUR CODE HERE
        self.dLdx = dLdy * self.sigmoid_func(self.X) * (1 - self.sigmoid_func(self.X))
        return self.dLdx

    def step(self, learning_rate=0):
        pass


class NLLLoss:
    def __init__(self):
        """
        Applies Softmax operation to inputs and computes NLL loss
        """
        # YOUR CODE HERE
        # (Hint: No code is expected here, just joking)
        pass

    def forward(self, X, y):
        """
        Passes objects through this layer.
        X is np.array of size (N, C), where C is the number of classes
        y is np.array of size (N), contains correct labels
        """
        self.L = -X[range(np.shape(X)[0]), y] + np.log(np.sum(np.exp(X), axis=1))
        self.X = X
        self.y = y
        # YOUR CODE HERE
        # Apply layer to input
        return self.L

    def backward(self, dLdy=0):
        """
        Note that here dLdy = 1 since L = y
        1. Compute dLdx
        2. Return dLdx
        """
        # YOUR CODE HERE
        self.dLdx = np.exp(self.X) / np.tile(np.sum(np.exp(self.X), axis=1).reshape(np.shape(self.X)[0], 1),
                                             (1, np.shape(self.X)[1]))
        self.dLdx[range(np.shape(self.X)[0]), self.y] -= 1
        return self.dLdx
        pass

    def step(self, learning_rate=0):
        pass

#%%
