# importing required modules
import os
import fnmatch
import fitz
import re

text = ''
for file in os.listdir('books'):
    if fnmatch.fnmatch(file, '*.pdf'):
        document = os.path.join('books', file)
        with fitz.open(document) as doc:
            for page in doc:
                text += page.getText()

# words = text.split(' ')
words = re.findall(r"[\w']+", text)
print(len(words))
# regex /[^a-zA-Za-åa-ö-w]/gi
regex = re.compile(r"/[^a-zA-Za-åa-ö-w]/gi")
words = list(dict.fromkeys(words))
print(len(words))
# new_list_of_words = [i for i in words if not regex.search(i)]
# print(len(new_list_of_words))
# print(new_list_of_words)
