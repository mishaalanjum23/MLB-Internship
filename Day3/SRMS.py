import json

# loading existing records automatically when program starts
try:
  with open("students.json", "r") as f:
     studentslist = json.load(f)

except:
    studentslist = []

while True:
    print("STUDENT RECORD MANAGEMENT SYSTEM")
    print("1. Add Student")
    print("2. View All Students")
    print("3. Search Student")
    print("4. Update Student")
    print("5. Delete Student")
    print("6. Exit")

    try:
        choice = int(input("Enter your choice: "))

        # add student
        if choice == 1:
            Students = {
             "name" : input("Enter student name:"),
             "Roll_no" : int(input("Enter roll-no:")),
             "age" : int(input("Enter student's age:")),
             "course" : input("Enter course name:")
            }
            studentslist.append(Students)
            print("student added successfully")

            # save student records to a json file
            with open("students.json", "w") as file:
             json.dump(studentslist, file)

        # view all students
        elif choice == 2:
            if len(studentslist) == 0:
             print("No students exist")
            else:
             for student in studentslist:
                 print(student)

        # search student
        elif choice == 3:
         id = int(input("Enter student's roll-no:"))
         found = False
         for student in studentslist:
             if student["Roll_no"] == id:
                 print(student)
                 found = True

         if found == False:
             print("Student is not found")

        #update student
        elif choice == 4:
            id = int(input("Enter roll-no to update:"))
            found = False
            for student in studentslist:
                if student["Roll_no"] == id:
                 student["name"] = input("Enter new name: ")
                 student["age"] = int(input("Enter new age: "))
                 student["course"] = input("Enter new course: ")

                 print("student updated successfully")
                 found = True

            if found == False:
             print("Student is not found")

            # to ensure changes are saved
            with open("students.json", "w") as file:
              json.dump(studentslist, file)

        # delete srudent
        elif choice == 5:
            id = int(input("Enter roll-no to delete: "))
            found = False
            for student in studentslist:
                if student["Roll_no"] == id:
                 studentslist.remove(student)
                 print("student deleted successfully")
                 found = True

            if found == False:
             print("Student is not found")

            with open("students.json", "w") as file:
             json.dump(studentslist, file)

        # Exit
        elif choice == 6:
            print("Program has ended")
            break

        else:
            print("Invalid choice")

    except:
        print("Invalid input")