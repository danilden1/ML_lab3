from sklearn import preprocessing
from sklearn.datasets import load_iris

# Загрузим данные из sklearn
iris = load_iris()
iris_data = iris.data
iris_target = iris.target

print(iris_target)