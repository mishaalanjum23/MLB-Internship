student = {
  "std_name" : input("Name:"),
  "std_class" : int(input("Class:")),
  "subjects" : [],
  "marks" : []
}

count = int(input("how many subjects?"))
for i in range(count):
    sub = input("Subject name:")
    score = int(input("subject marks:"))
    student["subjects"].append(sub)
    student["marks"].append(score)

average = sum(student["marks"])/len(student["marks"]) 

if average >= 80:
    grade = "A"
elif average >= 60:
    grade = "B"
elif average >= 40:
    grade = "C"
else:
    grade = "F" 

print("your grade is", grade)