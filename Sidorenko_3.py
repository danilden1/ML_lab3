import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
pd.set_option('display.max_columns', None)


dataset = pd.read_csv('AB_NYC_2019.csv')
print(dataset.info())

list_head = dataset.columns
print(list_head)

data = dataset.dropna()

for i in list_head:
    if dataset[i].dtype == object:
        print(i, ": ")
        print(dataset[i].unique())
        print(dataset[i].value_counts())

print(dataset.isnull().sum())
print(data.isnull().sum())

attributes = ["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",\
              "number_of_reviews", "last_review", "reviews_per_month"]

scatter_matrix(data[attributes])


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

scatter_matrix(data[attributes])
#delete hostels
data = data[(data[["calculated_host_listings_count"]] < 50).all(axis=1)]
print(data.describe())


scatter_matrix(data[attributes])


from sklearn.preprocessing import OneHotEncoder
data_cat = data["neighbourhood_group"]
data_cat_encoder, data_categories = data_cat.factorize()

print(data_cat_encoder[:10])

encoder = OneHotEncoder()
data_cat_1hot = encoder.fit_transform(data_cat_encoder.reshape(-1, 1))

print(data_cat_1hot.toarray())



plt.show()