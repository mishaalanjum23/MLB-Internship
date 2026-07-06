import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"c:\Users\sunny\OneDrive\Desktop\MLB-Internship\Day5\cleaned_student_performance.csv")

#	How many students are in the dataset?
print(len(df))

#	What is the average score for each subject?
subject_avg = df[["Maths_marks","science_marks","english_marks","SocialStudies_marks","language_marks"]].mean()

#	Who are the Top 5 performing students?
top5 = df.sort_values("Average_Score", ascending = False).head()
print("Top5 students are:", top5[["student_name", "Average_Score"]])

#  Which students need improvement?
std = df[df["Performance"] == "Needs Improvement"]
print("Students who need improvement are: ", std[["student_name", "Average_Score"]])

#  Which subject has the highest class average?
highest_sub_avg = subject_avg.max()
highest_sub_name = subject_avg.idxmax()
print("The subject with highest class average is", highest_sub_name, "with average score of ", highest_sub_avg) 

# charts:
# subject average:
plt.figure(figsize=(10,5))
plt.bar(subject_avg.index, subject_avg.values)
plt.title("Average Scores of Subjects")
plt.xlabel("Subjects")
plt.ylabel("Average Marks")
plt.xticks(rotation = 15)
plt.savefig("Subject_average.png")
plt.show()

# top 5 students
plt.figure(figsize=(10,5))
plt.bar(top5["student_name"], top5["Average_Score"])
plt.title("Top 5 Students")
plt.xlabel("Student Name")
plt.ylabel("Average Score")
plt.xticks(rotation = 15)
plt.savefig("Top_students.png")
plt.show()

# performance graph
performance_count = df["Performance"].value_counts()
plt.figure(figsize=(10,5))
plt.pie(performance_count.values, labels = performance_count.index, autopct = "%1.1f%%")
plt.title("Student Performance Distribution")
plt.savefig("Performance.png")
plt.show()