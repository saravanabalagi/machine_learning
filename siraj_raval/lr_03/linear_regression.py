import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression


class LinearRegression:
    def __init__(self, learning_rate, iterations):
        self.b = 0
        self.m = 0
        self.learning_rate = learning_rate
        self.iterations = iterations

    def get_y(self, x):
        return self.m * float(x) + self.b

    def get_total_error(self, x_values, y_values):
        error = 0
        for index, y in enumerate(y_values.ix[:, 0]):
            error += (float(y) - self.get_y(x_values.iloc[index][0])) ** 2
        return error / len(y_values.ix[:, 0])

    def step_gradient(self, x_values, y_values):
        print()
        print("Values before: m =", self.m, " b =", self.b)

        m_gradient = 0
        b_gradient = 0
        N = float(len(x_values.ix[:, 0]))

        print('%11s' % "d(m)", '%11s' % "m_gradient", '%11s' % "d(b)", '%11s' % "b_gradient")

        for i in range(int(N)):
            x = x_values.iloc[i][0]
            y = y_values.iloc[i][0]

            pm = (y - self.get_y(x)) * x
            pb = (y - self.get_y(x)) * -1

            m_gradient += pm * 2 / N
            b_gradient += pb * 2 / N

            print('%11s' % pm, '%11s' % m_gradient, '%11s' % pb, '%11s' % b_gradient)

        self.m -= self.learning_rate * m_gradient
        self.b -= self.learning_rate * b_gradient

        print("Values after: m =", self.m, " b =", self.b)
        print()

    def fit(self, x_values, y_values):
        for i in range(self.iterations):
            self.step_gradient(x_values, y_values)
        return

    def predict(self, x_values):
        predictions = []
        for x in x_values.ix[:, 0]:
            predictions.append(self.get_y(x))
        return predictions

dataframe = pd.read_csv('sample.csv', header=None)

x_values = dataframe[[0]]
y_values = dataframe[[1]]

x_values.columns = [0]
y_values.columns = [0]

clf = LinearRegression(0.0001, 1)
clf.fit(x_values, y_values)

# plt.scatter(x_values, y_values)
# plt.plot(x_values, clf.predict(x_values))
# plt.show()

