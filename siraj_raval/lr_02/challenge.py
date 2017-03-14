import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

dataframe = pd.read_csv('challenge_dataset.txt', header=None)

x_values = dataframe[[0]]
y_values = dataframe[[1]]

clf = linear_model.LinearRegression()
clf.fit(x_values, y_values)

plt.scatter(x_values, y_values)
plt.plot(x_values, clf.predict(x_values))
plt.show()



