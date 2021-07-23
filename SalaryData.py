import pandas as pd
import matplotlib.pyplot as plt

# create pandas dataframes from train, test datasets
train = pd.read_csv("SalaryData_Train.csv")
test = pd.read_csv("SalaryData_Test.csv")
print(train.info())

plt.bar(train.groupby('age').hoursperweek.mean().index, train.groupby('age').hoursperweek.mean())
plt.title("Average Hours per week by age")
plt.show()

plt.bar(train.groupby('age').capitalgain.mean().index, train.groupby('age').capitalgain.mean())
plt.title("Average capitalgain by age")
plt.show()

plt.bar(train.groupby('age').capitalloss.mean().index, train.groupby('age').capitalloss.mean())
plt.title("Average capitalloss by age")
plt.show()

# Defining the target variable
y_train = train.iloc[:, -1:]
y_test = test.iloc[:, -1:]
# print(y_test)

# identify numerical and categorical columns in input data
import numpy as np
numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train.select_dtypes('object').columns.tolist()[:-1] # excluding the target feature
# print(categorical_cols)

# Check the no. of categories for each categorical columns in test data
# print(test[categorical_cols].nunique())

# Check if there are any missing values # our data has no null values
# print(train.isna().sum())
# print(test.isna().sum())

# scaling numeric features
# print(train[numeric_cols].describe())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.fit_transform(test[numeric_cols])
# print(test)

# Encoding Categorical variables
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

encoder.fit(train[categorical_cols])
encoded_cols = list(encoder.get_feature_names(categorical_cols))
train[encoded_cols] = encoder.transform(train[categorical_cols])

encoder.fit(test[categorical_cols])
encoded_cols = list(encoder.get_feature_names(categorical_cols))
test[encoded_cols] = encoder.transform(test[categorical_cols])
# print(test[encoded_cols])


# Label encoding the target feature variables
y_train_con = pd.get_dummies(y_train, drop_first=True)
y_test_con = pd.get_dummies(y_test, drop_first=True)
# print(y_train_con.values.ravel())

# Model Building
from sklearn.svm import SVC
model1= SVC(kernel ='linear',random_state=0 )
model1.fit(train[numeric_cols + encoded_cols], y_train_con.values.ravel())
y_pred = model1.predict(test[numeric_cols + encoded_cols])

# Check the accuracy
print("Accuracy with Linear Kernel: ", model1.score(test[numeric_cols + encoded_cols], y_test_con.values.ravel())) # achieved 83.6% accuracy

from sklearn.metrics import confusion_matrix, classification_report
print("Confusion matrix: \n",confusion_matrix(y_test_con.values.ravel(),y_pred))
print('classification report: \n',classification_report(y_test_con.values.ravel(),y_pred))
