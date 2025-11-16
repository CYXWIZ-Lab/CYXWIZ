# Python Syntax Highlighting Test
# This file tests if all Python syntax is properly highlighted

# Keywords should be highlighted
def calculate_sum(a, b):
    """This is a docstring"""
    return a + b

# Variables and assignments
x = 10
y = 20.5
name = "CyxWiz"
enabled = True
disabled = False
nothing = None

# Control flow
if x > 5:
    print("x is greater than 5")
elif x == 5:
    print("x equals 5")
else:
    print("x is less than 5")

# Loops
for i in range(10):
    print(f"Iteration {i}")

while x > 0:
    x -= 1

# List comprehension
numbers = [1, 2, 3, 4, 5]
squared = [n**2 for n in numbers if n % 2 == 0]

# Function call with f-string
result = calculate_sum(10, 20)
print(f"Result: {result}")

# Try-except
try:
    import math
    print(f"Pi = {math.pi}")
except ImportError as e:
    print(f"Error: {e}")

# Class definition
class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

# Comments should be green/gray
# TODO: This is a comment
