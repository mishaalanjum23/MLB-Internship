import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"c:\Users\sunny\OneDrive\Desktop\MLB-Internship\Day6\student_performance.csv")

#encoding categorical columns 
le = LabelEncoder()
df["Program"] = le.fit_transform(df["Program"]) 
print(df["Program"])

#creating average score column
df["Average_score"] = df[["Python", "Mathematics" ,"Statistics", "Machine_Learning"]].mean(axis = 1)
df.to_csv("newfile.csv", index = False)

# Select appropriate feature columns (X) and target column (y).
x = df[["Attendance", "Program"]]
y = df["Average_score"] 

# Split the dataset into Training (80%) and Testing (20%).
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# create model 
model = LinearRegression()

#Train model
model.fit(x_train, y_train)

# making Prediction
predicted_y = model.predict(x_test)

#comparing actual with predicted values
table = pd.DataFrame({
    "Actual values" : y_test,
    "Predicted values" : predicted_y
})
print(table)

# Evaluating the results
mae = mean_absolute_error(y_test, predicted_y)
print( "Mean absolute error is", mae)

mse = mean_squared_error(y_test, predicted_y)
print("Mean squared error is", mse)

r2 = r2_score(y_test, predicted_y)
print("R2 Score", r2) 