import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#loading and preprocessing the dataset
breastcancer = load_breast_cancer()

X = breastcancer.data
y = breastcancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Hyperparameter tuning
model2 = LogisticRegression()
parameters ={
    "C" : [0.1, 1, 5, 10],
    "solver" : ["lbfgs", "liblinear"]
}

grid = GridSearchCV(estimator = model2, param_grid = parameters, cv = 5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y2_pred = best_model.predict(X_test)

print("Best parameters:", grid.best_params_)

cm2 = confusion_matrix(y_test, y2_pred)
acc = accuracy_score(y_test, y2_pred)
prec = precision_score(y_test, y2_pred)
rc = recall_score(y_test, y2_pred)
f1score = f1_score(y_test, y2_pred)

# Evaluation comparison 
print("\nBaseline model:")
print("\nConfusion Matrix:\n", cm)
print("\nAccuracy:", accuracy)
print("\nPrecision:", precision)
print("\nRecall:", recall)
print("\nF1 Score:", f1)

print("\nTuned model:")
print("\nConfusion Matrix:\n", cm2)
print("\nAccuracy:", acc)
print("\nPrecision:", prec)
print("\nRecall:", rc)
print("\nF1 Score:", f1score) 

#show confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.savefig("Baseline confusion matrix.png")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y2_pred)
plt.savefig("Tuned confusion matrix.png")
plt.show()