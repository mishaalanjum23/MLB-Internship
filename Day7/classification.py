from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#dataset
iris = load_iris()

#features n target
X = iris.data 
y = iris.target 

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# creating and training a logistic regression model
model = LogisticRegression()

model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test) 

# Evaluating the model 
confusionmatrix = confusion_matrix(y_test, y_pred) 
print("Confusion Matrix:", confusionmatrix)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred, average = "weighted")
print("Precision:", precision)

recall = recall_score(y_test, y_pred, average = "weighted")
print("Recall:", recall)

f1 = f1_score(y_test, y_pred, average = "weighted")
print("f1:", f1)
