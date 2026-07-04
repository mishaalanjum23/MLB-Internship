# 1. Create a text file and write data into it.

with open("introduction.txt", "w") as file:
    file.write("Hi, my name is Mishaal")


# 2. Read and display file contents.
with open("introduction.txt", "r") as file:
    content = file.read()
    print(content)


# 3. Append new data to an existing file.
with open("introduction.txt", "a") as file:
    file.write("\nI am 23 years old")


# 4. Count the number of lines in a file.
with open("introduction.txt", "r") as file:
    num = 0
    for line in file:
     num += 1
    print(num)
    