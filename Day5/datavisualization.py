import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"c:\Users\sunny\OneDrive\Desktop\MLB-Internship\Day5\cleaned_student_performance.csv")

# •	Bar chart showing the average marks of each student.
plt.figure()
plt.bar(df["student_name"], df["Average_Score"])
plt.title("Average Marks of Each Student")
plt.xlabel("Students")
plt.ylabel("Average Score")
plt.xticks(rotation=90)
plt.show() 

# •	Histogram showing the distribution of Average Scores.
plt.figure()
plt.hist(df["Average_Score"], bins = 10)
plt.title("Distribution of Average Scores")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")
plt.show()

#	Scatter plot comparing any 2 subjects marks.
plt.figure()
plt.scatter(df["Maths_marks"], df["english_marks"])
plt.title("Maths vs English Marks")
plt.xlabel("Maths Marks")
plt.ylabel("English Marks")
plt.show()

#  Pie chart showing the distribution of students by Performance category. 
plt.figure()
performance = df["Performance"].value_counts()
plt.pie(performance, labels = performance.index, autopct = "%1.1f%%")
plt.title("Performance Distribution")
plt.show()

#•	Box plot to visualize the spread of marks in all subjects.
plt.figure()
plt.boxplot(df[["Maths_marks","science_marks","english_marks","SocialStudies_marks","language_marks"]])
plt.title("Marks Distribution")
plt.xticks([1,2,3,4,5],
           ["Maths","Science","English","Social Studies","Language"])
plt.show()