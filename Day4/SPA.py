import pandas as pd
# Load the dataset using Pandas.
df = pd.read_csv(r"c:\Users\sunny\OneDrive\Desktop\MLB-Internship\Day4\StudentMarks.csv")

# Display basic information about the dataset.
df.info()

# Calculate average marks for each subject.
print("Average marks:")
print(df[["maths_marks", "science_marks", "english_marks", "social_studies_marks", "language_marks"]].mean())

# Create total marks column
df["Total"] = df[["maths_marks", "science_marks", "english_marks", "social_studies_marks", "language_marks"]].sum(axis=1)

# Identify the top 5 performing students.
top5 = df.sort_values(by="Total", ascending=False).head(5)
print("\nTop 5 Students:", top5)

# Find students scoring below the average.
average_total = df["Total"].mean()
below_average = df[df["Total"] < average_total]
print("\nStudents below average are:", below_average)

# Display the total number of students.
print("Total students:", len(df))

df.to_csv("student_report.csv", index = False)