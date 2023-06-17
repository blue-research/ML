import re

text = input("Enter a string: ")

if re.search(r'tesla|(first.*republic)', text, re.IGNORECASE):
    print("Match found!")
else:
    print("No match found.")