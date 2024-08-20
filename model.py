# import section
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# load data
train_housing = pd.read_csv('dataset/train.csv')
test_housing = pd.read_csv('dataset/test.csv')

# combine train and test dataset
# axis = 0 -> concat along rows
data = pd.concat([train_housing, test_housing], axis = 0)

print(data)

# clean the data (by define some values or just remove it)
# first, find feature(s) with more than 1000 NULL values
features = []
nullValues = []

for i in data:
    if (data.isna().sum()[i]) > 1000 and i!= 'SalePrice':
        features.append(i)
        nullValues.append(data.isna().sum()[i])

y_pos = np.arange(len(features))
plt.bar(y_pos, nullValues, align = 'center', alpha = 1.0) # alpha = Opacity
plt.xticks(y_pos, features, rotation=45, ha='right')
plt.ylabel('NULL values')
plt.xlabel('Features')
plt.title('feature(s) with more than 1k NULL values')
# Add value labels on top of the bars
for i in range(len(nullValues)):
    plt.text(i, nullValues[i] + 50, str(nullValues[i]), ha='center')
plt.show()

