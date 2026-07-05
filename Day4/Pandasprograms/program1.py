import pandas as pd

# 1. Load the dataset.
df = pd.read_csv(r"C:\Users\sunny\OneDrive\Desktop\MLB-Internship\Day4\Pandasprograms\students.csv")

# 2. Display the first and last five rows.
firstfive_rows = df.head()
lastfive_rows = df.tail()
print(firstfive_rows)
print(lastfive_rows)

# 3. Display dataset information. 
df.info()

# 4. Find missing values. 
print(df.isnull().sum()) 

# 5. Filter data based on a condition. 
print(df[df["Age"] >= 22])

# 6. Calculate summary statistics.
print(df.describe())