import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# step 1: Load and explore the dataset
breastcancer = load_breast_cancer()
df = pd.DataFrame(data = breastcancer.data, columns = breastcancer.feature_names)

print(df.head())
print(df.describe()) 
df.info()

df["target"] = breastcancer.target
print(df["target"].value_counts())

# step 2: Build a baseline model
X = breastcancer.data
y = breastcancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Baseline Confusion Matrix:", cm)

accuracy = accuracy_score(y_test, y_pred)
print("Baseline Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print("Baseline Precision:", precision)

recall = recall_score(y_test, y_pred)
print("Baseline Recall:", recall)

f1 = f1_score(y_test, y_pred)
print("Baseline F1 Score:", f1)

# step 3: Hyperparameter tuning
model2 = LogisticRegression()
parameters ={
    "C" : [0.1, 1, 5, 10],
    "solver" : ["lbfgs", "liblinear"]
}

grid = GridSearchCV(estimator = model2, param_grid = parameters, cv = 5)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

best_model = grid.best_estimator_
y2_pred = best_model.predict(X_test)

cm2 = confusion_matrix(y_test, y2_pred)
print("Tuned confusion matrix:", cm2)

acc = accuracy_score(y_test, y2_pred)
print("Tuned accuracy:", acc)

prec = precision_score(y_test, y2_pred)
print("Tuned precision:", prec)

rc = recall_score(y_test, y2_pred)
print("Tuned recall:", rc)

f1score = f1_score(y_test, y2_pred)
print("Tuned f1 score:", f1score)