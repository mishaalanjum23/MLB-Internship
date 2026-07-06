import pandas as pd

df = pd.read_csv(r"c:\Users\sunny\OneDrive\Desktop\MLB-Internship\Day5\StudentMarks.csv")

# Check for missing values. 
print(df.isnull().sum())

# Remove duplicate records (if any).
print(df.duplicated())

# Rename one or more columns. 
df = df.rename(columns = {
    "maths_marks" : "Maths_marks",
    "social_studies_marks" : "SocialStudies_marks"
})

# Create a new column named Average_Score by calculating the average marks across all subjects.
df["Average_Score"] = df[["Maths_marks","science_marks","english_marks","SocialStudies_marks","language_marks"]].mean(axis=1)

# Creating performance column
def performance(avg_marks):
    if avg_marks  >= 90:
        return("Excellent")
    elif avg_marks >= 80 and avg_marks <= 89:
        return("Good")
    elif avg_marks >= 70 and avg_marks <= 79:
        return("Average")
    elif avg_marks < 70:
        return("Needs Improvement")
    
df["Performance"] = df["Average_Score"].apply(performance)

# saving dataset
df.to_csv("cleaned_student_performance.csv", index = False)