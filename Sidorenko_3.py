import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
pd.set_option('display.max_columns', None)


dataset = pd.read_csv('AB_NYC_2019.csv')
print(dataset.info())
#list of heads
list_head = dataset.columns
print(list_head)
#delete zeroas values
data = dataset.dropna()
#show all uniqe values from object
for i in list_head:
    if dataset[i].dtype == object:
        print(i, ": ")
        print(dataset[i].unique())
        print(dataset[i].value_counts())

print(dataset.isnull().sum())
print(data.isnull().sum())
#----------add new collums to make room type numbers--------
unique_type_of_room = data['room_type'].unique()
print(unique_type_of_room)
index_for_unique_type_of_room = [2, 3, 1]


def get_index(line, index_mas):
    for i in range(len(unique_type_of_room)):
        if line == unique_type_of_room[i]:
            return index_mas[i]


data.loc[:, 'type_of_room'] = data['room_type'].apply(lambda x: get_index(str(x), index_for_unique_type_of_room))


#attributes for histogram analice

attributes = ["room_type", "price", "minimum_nights","type_of_room",\
              "number_of_reviews", "last_review", "reviews_per_month", "availability_365"]

#scatter_matrix(data[attributes])


print(data.describe())

a_series = (data != 0).any(axis=1)
new_df = data.loc[a_series]
#delete zeros price
data = data[(data[["price"]] != 0).all(axis=1)]
data = data[(data[["price"]] < 800).all(axis=1)]
#delete review per month mo 30
data = data[(data[["reviews_per_month"]] < 30).all(axis=1)]
#delete minimum nighte
data = data[(data[["minimum_nights"]] < 29).all(axis=1)]
print(data.describe())

#delete hostels
data = data[(data[["calculated_host_listings_count"]] < 50).all(axis=1)]
print(data.describe())


#scatter_matrix(data[attributes])


from sklearn.preprocessing import OneHotEncoder
data_cat = data["neighbourhood_group"]
data_cat_encoder, data_categories = data_cat.factorize()

encoder = OneHotEncoder()
data_cat_1hot = encoder.fit_transform(data_cat_encoder.reshape(-1, 1))

print(data_cat_1hot.toarray())

print(data_cat_1hot.size)

data.loc[:, ['name1', 'name2', 'name3', 'name4', 'name5']] = data_cat_1hot.toarray()


attributes2 = ["minimum_nights", "type_of_room",  \
              "number_of_reviews", "availability_365",  \
               'name1', 'name2', 'name3', 'name4', 'name5']

#scatter_matrix(data[attributes2])

#plt.show()
list_head_ml = data.columns

from sklearn import preprocessing
min_max_skaler = preprocessing.MinMaxScaler()
#data = min_max_skaler.fit_transform(data)

data_num = data[attributes2]

#делаем из цен группы
labels = ['free', 'cheap', 'a little more expensive', 'average', 'expensive']
price_group = pd.cut(data['price'],
                    bins=[0, 50, 100, 300, 500, data['price'].max()],
                    labels=labels)

data_price = price_group

list_ml_num = data_num.columns
print(list_ml_num)

from sklearn.decomposition import PCA

# Метод главных компонент
pca = PCA(n_components=2)
pca.fit(data_num)
data_reduced = pca.transform(data_num)
#нормализация
data_reduced = min_max_skaler.fit_transform(data_reduced)

from matplotlib import cm
import numpy
fig = plt.figure()
fig.suptitle("Scatter plot normalize")
labels = attributes2
plots = []
colors = cm.rainbow(numpy.linspace(0, 1, len(data_price.unique())))

for ing, ng in enumerate(data_price.unique()):
    print("ing = ", ing, "  ng = ", ng)
    print("data_price.unique()[ing] = ", data_price.unique()[ing])
    plots.append(plt.scatter(x=data_reduced[data_price == ng, 0],
                            y=data_reduced[data_price == ng, 1],
                            c=colors[ing],
                            edgecolor='k'))

plt.xlabel("component1")
plt.ylabel("component2")
plt.legend(plots, data_price.unique(), loc="lower right", title="species")
plt.show()

"""
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data_reduced, test_size=0.2, random_state=42)

train_set_price, test_set_price = train_test_split(data_reduced_price, test_size=0.2, random_state=42)

#X = train_set[["minimum_nights", "type_of_room", "reviews_per_month", "availability_365", 'name1', 'name2', 'name3', 'name4', 'name5']]
X = train_set_price
list_ml = X.columns

Y = train_set

#X_new = test_set[["minimum_nights", "type_of_room", "reviews_per_month", "availability_365", 'name1', 'name2', 'name3', 'name4', 'name5']]
X_new = test_set_price

Y_new = test_set
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
lin_reg = Ridge(alpha=1, solver="cholesky")
lin_reg.fit(X, Y)

y_pred = lin_reg.predict(X_new)

df = pd.DataFrame({'Actual': Y_new, 'Predicted': y_pred})

print("result: ")
print(df)

plt.figure(num="name")
plt.plot(X, Y, "b.")
plt.plot(X_new, y_pred, "r-")


plt.show()

#print(lin_reg.intercept_, lin_reg.coef_)
#print(lin_reg.predict(X_new))
#print(Y_new[:50])
"""