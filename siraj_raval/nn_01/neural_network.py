from numpy import dot, exp, random, array


class NeuralNetwork:
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1
        self.iterations = 10

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, x_values, y_values, iterations):
        for i in range(iterations):
            output = self.predict(x_values)
            error = y_values - output
            adjustment = dot(x_values.T, error * self.__sigmoid_derivative(output))

            print("y_values:", y_values)
            print("Output:", output)
            print("Error:", error)
            print("x_values dot Error:", dot(x_values.T, error))
            print("Gradient:", error * self.__sigmoid_derivative(output))
            print("X-Values.T:", x_values.T)
            print("Adjustment:", adjustment)

            self.synaptic_weights += adjustment
        return

    def predict(self, x_values):
        predictions = []
        for x in x_values:
            prediction = self.__sigmoid(dot(x, self.synaptic_weights))
            predictions.append(prediction)
        return array(predictions)


neural_network = NeuralNetwork()

x_values = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
y_values = array([[0, 1, 1, 0]]).T

neural_network.fit(x_values, y_values, 2)
print(neural_network.predict([[1, 0, 0]]))
