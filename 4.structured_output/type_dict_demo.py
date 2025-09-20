from typing import TypedDict

# define a structure of dictionary, but it does not give error if you chage datatype
class Person(TypedDict):

    name: str 
    age: int


new_person: Person = {'name':'spidy', 'age': 25}

print(new_person)