import pandas as pd
from urllib.request import urlretrieve


DOWNLOAD = False

# download data from : http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
if DOWNLOAD:
    data_path, _ = urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", "car.csv")

# use pandas to view the data structure
col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
data = pd.read_csv("car.csv", names=col_names)
print(data.head())
print("\nNum of data: ", len(data))     # 1728


# view data values
for name in col_names:
    print(name, pd.unique(data[name]))


# covert data to onehot representation
new_n_col = sum([len(pd.unique(data[name])) for name in col_names])
new_data = pd.get_dummies(data, prefix=col_names)
print("\n", new_data.head(3))

# save and view
new_data.to_csv("car_onehot.csv", index=False)