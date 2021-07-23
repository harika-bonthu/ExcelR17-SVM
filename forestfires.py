# import necessary modules
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# creating a dataset from forestfires.csv file
data = pd.read_csv("forestfires.csv")

# Plotting average of FFMC, DMC, DC, ISI, RH by grouping months and days.
plt.bar(data.groupby('month').FFMC.mean().index, data.groupby('month').FFMC.mean())
plt.bar(data.groupby('month').DMC.mean().index, data.groupby('month').DMC.mean())
plt.bar(data.groupby('month').ISI.mean().index, data.groupby('month').ISI.mean())
plt.title("Average FFMC, DMC, ISI of each month")
plt.legend(['FFMC', 'DMC', 'ISI'])
plt.show()

plt.bar(data.groupby('month').DC.mean().index, data.groupby('month').DC.mean())
plt.bar(data.groupby('month').RH.mean().index, data.groupby('month').RH.mean())
plt.title("Average DC, RH of each month")
plt.legend(['DC', 'RH'])
plt.show()

plt.bar(data.groupby('day').DMC.mean().index, data.groupby('day').DMC.mean())

plt.bar(data.groupby('day').RH.mean().index, data.groupby('day').RH.mean())
plt.title("Average DMC, RH of each day")
plt.legend(['DMC', 'RH'])
plt.show()

plt.bar(data.groupby('day').FFMC.mean().index, data.groupby('day').FFMC.mean())
plt.bar(data.groupby('day').ISI.mean().index, data.groupby('day').ISI.mean())
# plt.bar(data.groupby('day').DC.mean().index, data.groupby('day').DC.mean())
plt.title("Average FFMC, ISI of each day")
plt.legend(['FFMC', 'ISI'])
plt.show()

# droppping unnecessary columns
mydata = data.drop(['month', 'day'], axis=1)
# print(mydata.head())

# check if there are any null values
# print(mydata.isna().sum()) 

# Defining independent and dependent variables
X = mydata.iloc[:, :-1]
y = mydata.iloc[:, -1]

# create dummy variables for the dependent feature
y = pd.get_dummies(y, drop_first=True)
# print(y)

# Normalizing the dependent variable
normalizer = Normalizer()
X_norm = normalizer.fit_transform(X)
# print(X_norm)

# splitting data into train, test datasets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)

# instantiating SVC object with linear kernel
model1 = SVC(kernel='linear', random_state=0)

# training the model
model1.fit(X_train, y_train)

# predict
y_pred1 = model1.predict(X_test)

# Check the accuracy
print("Accuracy with Linear Kernel: ", model1.score(X_test, y_test)) # achieved 83.6% accuracy
print("Confusion matrix: \n",confusion_matrix(y_test,y_pred1))
print('classification report: \n',classification_report(y_test,y_pred1))

# instantiating SVC object with RBF kernel
model2 = SVC(kernel='rbf', random_state=0)

# training the model
model2.fit(X_train, y_train)

# predict
y_pred2 = model2.predict(X_test)

# Check the accuracy
print("Accuracy with RBF Kernel: ", model2.score(X_test, y_test)) # achieved 85.5% accuracy
print("Confusion matrix: \n",confusion_matrix(y_test,y_pred2))
print('classification report: \n',classification_report(y_test,y_pred2))

# instantiating SVC object with Polynomial kernel
model3 = SVC(kernel='poly', random_state=0)

# training the model
model3.fit(X_train, y_train)

# predict
y_pred3 = model3.predict(X_test)

# Check the accuracy
print("Accuracy with Polynomial Kernel: ", model3.score(X_test, y_test)) # achieved 86.5% accuracy
print("Confusion matrix: \n",confusion_matrix(y_test,y_pred3))
print('classification report: \n',classification_report(y_test,y_pred3))

# instantiating SVC object with Sigmoid kernel
model4 = SVC(kernel='sigmoid', random_state=0)

# training the model
model4.fit(X_train, y_train)

# predict
y_pred4 = model4.predict(X_test)

# Check the accuracy
print("Accuracy with Sigmoid Kernel: ", model4.score(X_test, y_test)) # achieved 80.7% accuracy
print("Confusion matrix: \n",confusion_matrix(y_test,y_pred4))
print('classification report: \n',classification_report(y_test,y_pred4))