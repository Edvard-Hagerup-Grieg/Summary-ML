import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ADALINE as adl
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

#слишком медленная сходимость
adln1 = adl.ADALINE()
adln1._init_(eta=0.0001, n_iter=10)
adln1.fit(X, y)
plt.plot(range(1, len(adln1.cost_) + 1), adln1.cost_, marker='o', color='b', alpha = 0.2)

#сходимости нет, т.к. с такой скоростью обучения перескочили глобальный минимум
adln2 = adl.ADALINE()
adln2._init_(eta=0.01, n_iter=10)
adln2.fit(X, y)
plt.plot(range(1, len(adln2.cost_) + 1), np.log10(adln2.cost_), marker='o', color='r', alpha = 0.2)

#сходимость при той же скорости обучения 0.01 после стандартизации
adln = adl.ADALINE()
adln._init_(eta=0.01, n_iter=10)
adln.fit(X_std, y)
plt.plot(range(1, len(adln.cost_) + 1), adln.cost_, marker='o', color='r')

plt.xlabel('Эпохи')
plt.ylabel('Сумма квадратичных ошибок')

plt.figure(2)
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
plot_decision_regions(X_std, y, classifier=adln)
plt.title('ADALINE (градиентный спуск)')
plt.xlabel('длина чашелистика [стандартизованная]')
plt.ylabel('длина лепестка [стандвртизованная]')
plt.legend(loc='upper left')

plt.figure(3)
plot_decision_regions(X_std, y, classifier=adln2)
plt.title('ADALINE (градиентный спуск)')
plt.xlabel('длина чашелистика [не стандартизованная]')
plt.ylabel('длина лепестка [не стандвртизованная]')
plt.legend(loc='upper left')

plt.show()
