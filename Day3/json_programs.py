# 1. Store student information in a JSON file.
import json

student = {
    "name" : "Mishaal",
    "roll_no" : 57,
    "degree" : "Software Engineering"
}

with open("student.json", "w") as file:
    json.dump(student, file) 


# 2. Read data from a JSON file.
with open("student.json", "r") as file:
    data = json.load(file)
    print(data)


# 3. Update an existing student's information.
with open("student.json", "r") as file:
    info = json.load(file)
    info["CGPA"] = 3.4
    with open("student.json", "w") as f:
     json.dump(info, f)


# 4. Add a new student to the JSON file.
student2 = {
       "name" : "Zaha",
       "roll_no" : 58 ,
       "degree" : "Software Engineering"
    }
 
with open("student.json", "r") as file:
    existing_student = json.load(file) 
    studentlist = []
    studentlist.append(existing_student)
    studentlist.append(student2)

    with open("student.json", "w") as f:
       json.dump(studentlist, f) 
