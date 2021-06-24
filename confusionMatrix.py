import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt  

df = pd.read_csv('BankNote_Authentication.csv')

print(df.head())

y = df['class']

X = df[['variance', 'skewness', 'curtosis', 'entropy']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

LR = LogisticRegression()
LR.fit(X_train,y_train) 

y_prediction = LR.predict(X_test) 

predicted_values = []
for i in y_prediction:
  if i == 0:
    predicted_values.append("Authorized")
  else:
    predicted_values.append("Forged")

actual_values = []
for i in y_test:
  if i == 0:
    actual_values.append("Authorized")
  else:
    actual_values.append("Forged")

y_prediction = LR.predict(X_test)

predicted_values = []

for i in y_prediction:
  if i == 0:
    predicted_values.append('No')
  else:
    predicted_values.append('Yes')

actual_values = []

for i in y_test.ravel():
  if i == 0:
    actual_values.append('No')
  else:
    actual_values.append('Yes')

labels = ["Yes", "No"]

cm = confusion_matrix(actual_values, predicted_values, labels)

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)

tn, fp, fn, tp = confusion_matrix(y_test, y_prediction).ravel()
print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Positives: ", tp)

accuracy = 150 + 189 / 150 + 189 + 2 + 2

accuracy = 339 / 343

print(accuracy)