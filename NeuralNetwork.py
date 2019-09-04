import numpy as np


class NeuralNetwork:
    def __init__(self, layers, lr=0.01):
        # By default we are going to use sigmoid activation function and bias too.
        self._layers = layers
        self._lr = lr

        self._layer_size = len(layers)

        # Right now didn't updated bias
        self._B = [0] * self._layer_size

        self._W = []

        for i in range(self._layer_size - 1):
            mat = np.random.randn(self._layers[i], self._layers[i + 1])
            self._W.append(mat)

        self._A = [0] * self._layer_size
        self._D = [0] * self._layer_size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X, Y, iter=100):
        for i in range(iter):
            output = self.forward(X)
            loss = self.backward(Y, output)
            print(f"Iteration: {i}, Loss: {loss}")

    def forward(self, X):
        # self._A[0] = self.sigmoid(X)
        self._A[0] = X  # This one gets better result than above.

        for i in range(self._layer_size - 1):
            self._A[i + 1] = self.sigmoid(np.dot(self._A[i], self._W[i]) + self._B[i])

        return self._A[self._layer_size - 1]

    def backward(self, Y, output):
        # Mean square error
        Y_delta = output - Y

        loss = np.mean(np.square(Y_delta))

        derivative = self.sigmoid_derivative(output)
        output_delta = Y_delta * derivative

        prev_delta = output_delta

        self._D[self._layer_size - 2] = output_delta

        for i in range(self._layer_size - 2, -1, -1):
            temp = np.dot(prev_delta, self._W[i].T)
            prev_delta = temp * self.sigmoid_derivative(self._A[i])
            self._D[i - 1] = prev_delta

        for i in range(self._layer_size - 1):
            self._W[i] = self._W[i] - self._lr * (np.dot(self._A[i].T, self._D[i]))
            # self._W[i] = self._W[i] - (np.dot(self._A[i].T, self._D[i]))

        return loss

    def sigmoid_derivative(self, o):
        return o * (1 - o)


if __name__ == '__main__':
    # XOR classification
    # X = np.array(([0, 0], [0, 1], [1, 0], [1, 1]), dtype=float)
    # Y = np.array(([0, 1], [1, 0], [1, 0], [0, 1]), dtype=float)
    #
    # nn = NeuralNetwork([2, 50, 50, 2], lr=0.1)
    # nn.train(X, Y, iter=1000)
    # print(nn.forward(X))

    # Classification on random dataset
    nn = NeuralNetwork([3, 50, 50, 3], lr=0.1)

    X = np.array(([0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 1]), dtype=float)
    Y = np.array(([1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]), dtype=float)

    nn.train(X, Y, iter=1000)

    print(nn.forward(X))
