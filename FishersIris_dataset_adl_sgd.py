import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ADALINE_sgd as adlsgd
from matplotlib.colors import ListedColormap


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header = None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

plt.figure(1)

adl = adlsgd.ADALINE_sgd()
adl._init_(eta=0.01, n_iter=15, random_state=1)
adl.fit(X_std, y)
plt.plot(range(1, len(adl.cost_) + 1), adl.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

def plot_decision_regions (X, y, classifier, resolution=0.02) :
    # настроить генератор маркеров и палитру
    markers = ('s', 'x', 'o','^','v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # вывести поверхность решения
    xl_min, xl_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xxl, xx2 = np.meshgrid(np.arange(xl_min, xl_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xxl.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xxl.shape)
    plt.contourf(xxl, xx2, Z, alpha=0.4 , cmap=cmap)
    plt.xlim(xxl.min(), xxl.max())
    plt.ylim(xx2.min(), xx2.max())

    # показать образцы классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

plt.figure(2)
plot_decision_regions(X_std, y, classifier=adl)
plt.title('ADALINE (stochastic gradient descent)')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.show()
