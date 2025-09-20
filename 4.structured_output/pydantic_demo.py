from pydantic import BaseModel, Field
from typing import Optional

## in this if you give some other data type then it will give an error
class Student(BaseModel):

    name: str = 'spidy'
    age: Optional[int] = None
    # data type = greater than, less than
    cgpa: float = Field(gt=0, lt=10, default=7, description='A decimal value representing the cgpa of the student')


new_student = {'age':24, 'email':'abc@gmail.com'}

student = Student(**new_student)

print(student)

# convert
student_dict = dict (student)

print(student_dict['age'])

student_json = student.model_dump_json()
print(student_json)