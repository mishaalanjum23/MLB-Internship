import pandas as pd
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data 
y = iris.target 

print("IRIS FLOWER CLASSIFICATION SYSTEM")

print("Features names:", iris.feature_names)
print("Target names:", iris.target_names)
print("Dataset size:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test) 

# Evaluation metrics
confusionmatrix = confusion_matrix(y_test, y_pred) 
print("Confusion matrix:", confusionmatrix)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred, average = "weighted")
print("Precision:", precision)

recall = recall_score(y_test, y_pred, average = "weighted")
print("Recall:", recall)

f1 = f1_score(y_test, y_pred, average = "weighted")
print("F1:", f1) 

# Actual vs predicted values
table = pd.DataFrame({
    "Actual values" : iris.target_names[y_test],
    "Predicted values" : iris.target_names[y_pred]
})
print(table)

#show confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig("confusion_matrix.png")
plt.show()