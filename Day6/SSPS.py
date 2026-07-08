import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv(r"c:\Users\sunny\OneDrive\Desktop\MLB-Internship\Day6\student_performance.csv")

# data preprocessing
le = LabelEncoder()
df["Program"] = le.fit_transform(df["Program"]) 

df["Average_score"] = df[["Python", "Mathematics" ,"Statistics", "Machine_Learning"]].mean(axis = 1)

x = df[["Attendance", "Program"]]
y = df["Average_score"] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

print("STUDENT SCORE PREDICTION SYSTEM")

# linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predicting average scores
predicted_y = model.predict(x_test)

#model evaluation
print("\n Model Evaluation Metrics")
mae = mean_absolute_error(y_test, predicted_y)
print( "\nMean absolute error:", mae)

mse = mean_squared_error(y_test, predicted_y)
print("\nMean squared error:", mse)

r2 = r2_score(y_test, predicted_y)
print("\nR2 Score:", r2)

# comparison table
print("\nActual vs Predicted scores")
table = pd.DataFrame({
    "Actual values" : y_test,
    "Predicted values" : predicted_y
})
print(table)

# scatter plot
plt.figure()
plt.scatter(y_test, predicted_y)
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.savefig("Actual_vs_predicted.png")
plt.show()