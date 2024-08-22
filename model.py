# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
train_housing = pd.read_csv('dataset/train.csv')
validation_housing = pd.read_csv('dataset/test.csv')

# Combine train and test dataset
data = pd.concat([train_housing, validation_housing], axis=0)
print(data)

# Identify and plot features with more than 1000 NULL values
features = []
nullValues = []

for i in data:
    if data.isna().sum()[i] > 1000 and i != 'SalePrice':
        features.append(i)
        nullValues.append(data.isna().sum()[i])

y_pos = np.arange(len(features))
plt.bar(y_pos, nullValues, align='center', alpha=1.0)
plt.xticks(y_pos, features, rotation=45, ha='right')
plt.ylabel('NULL values')
plt.xlabel('Features')
plt.title('Features with more than 1k NULL values')
plt.show()

# Deal with NULL values
data = data.dropna(axis=1, thresh=1000)
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
data = pd.get_dummies(data)  # Convert string values to 0/1

# Deal with correlations
covarianceMatrix = data.corr()
listOfFeatures = [i for i in covarianceMatrix]
setOfDroppedFeatures = set()

for i in range(len(listOfFeatures)):
    for j in range(i+1, len(listOfFeatures)):
        feature1 = listOfFeatures[i]
        feature2 = listOfFeatures[j]
        if abs(covarianceMatrix[feature1][feature2]) > 0.8:
            if abs(covarianceMatrix[feature1]["SalePrice"]) > abs(covarianceMatrix[feature2]["SalePrice"]):
                setOfDroppedFeatures.add(feature2)
            else:
                setOfDroppedFeatures.add(feature1)

data = data.drop(setOfDroppedFeatures, axis=1)

# Drop features not correlated with SalePrice
nonCorrelatedWithOutput = [column for column in data if abs(data[column].corr(data["SalePrice"])) < 0.045]
data = data.drop(nonCorrelatedWithOutput, axis=1)

# Plot SalePrice vs LotArea
plt.plot(data['LotArea'], data['SalePrice'], 'bo')
plt.axvline(x=75000, color='r')
plt.ylabel('SalePrice')
plt.xlabel('LotArea')
plt.title('SalePrice as a function of LotArea')
plt.show()

# Separate train and test sets
train_set = data.iloc[:1460]
test_set = data.iloc[1460:]

# Function to identify outliers using IQR
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))[0]

# Apply outlier detection only to numeric columns
numeric_columns = train_set.select_dtypes(include=[np.number]).columns

# Ensure 'Id' column is not processed
if 'Id' in numeric_columns:
    numeric_columns = numeric_columns.drop('Id')

# Drop outlier values from the train set
outlier_indices = set()
for column in numeric_columns:
    outlier_indices.update(outliers_iqr(train_set[column]))

trainWithoutOutliers = train_set.drop(index=outlier_indices)

# Train-Validation Split
X = trainWithoutOutliers.drop("SalePrice", axis=1)
Y = np.log1p(trainWithoutOutliers["SalePrice"])
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the model
reg = LinearRegression().fit(X_train, Y_train)

# Make predictions on validation set
Y_val_pred = reg.predict(X_val)

# Evaluate the model
mae = mean_absolute_error(Y_val, Y_val_pred)
mse = mean_squared_error(Y_val, Y_val_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_val, Y_val_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R²: {r2}')

# Train on the entire training set and make predictions on test set
X_full = trainWithoutOutliers.drop("SalePrice", axis=1)
Y_full = np.log1p(trainWithoutOutliers["SalePrice"])
reg_full = LinearRegression().fit(X_full, Y_full)

# Make predictions
test_set = test_set[X_full.columns]  # Ensure test_set has the same columns as X_full
pred = np.expm1(reg_full.predict(test_set))

# Load the answer file
actual_results = pd.read_csv('submission.csv')

# Compare predictions with actual results
# Ensure your predictions and the answer file have a common key (like 'Id')
try:
    sub = pd.DataFrame()
    sub['Id'] = validation_housing['Id']
    sub['myModelPredictedSalePrice'] = pred
    
    # Merge the predictions with actual results based on 'Id'
    comparison = pd.merge(sub, actual_results, on='Id', how='inner')
    
    # Calculate accuracy metrics
    mae = mean_absolute_error(comparison['PredictedSalePrice'], comparison['myModelPredictedSalePrice'])
    mse = mean_squared_error(comparison['PredictedSalePrice'], comparison['myModelPredictedSalePrice'])
    rmse = np.sqrt(mse)
    r2 = r2_score(comparison['PredictedSalePrice'], comparison['myModelPredictedSalePrice'])
    
    print(f'Final MAE: {mae}')
    print(f'Final MSE: {mse}')
    print(f'Final RMSE: {rmse}')
    print(f'Final R²: {r2}')
    
    sub.to_csv("submission_from_model.csv", index=False)
    print(sub)
except Exception as e:
    print("Error:", e)