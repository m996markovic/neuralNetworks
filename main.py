# importing required modules
import os
import fnmatch
import fitz
import re

import gensim
from gensim.models import Word2Vec

# Reading all PDF files from 'books' directory and storing it in text variable as string
text = ''
for file in os.listdir('books'):
    if fnmatch.fnmatch(file, '*.pdf'):
        document = os.path.join('books', file)
        with fitz.open(document) as doc:
            for page in doc:
                text += page.getText()

# Find all words in text and store it to the list of separate strings
words = re.findall(r"[\w']+", text)
# Here we have around 760103 words
# print(len(words))
new_words = []
for word in words:
    new_words.append(word.lower())
# Removing all duplicates from the list
new_words = list(dict.fromkeys(new_words))
# Now we have 89250 words
# print(len(new_words))

# Creating regex for matching only words written in Serbian Latin alphabet and removing those that do not match it
serbian_latin_alphabet = "abcdefghijklmnopqrstuvwxyzčćđšž"  # define the Serbian Latin alphabet
regex = f'^[{serbian_latin_alphabet}{serbian_latin_alphabet.upper()}]+$'
new_list_of_words = [i for i in new_words if re.match(regex, i)]
# print(len(new_list_of_words))
# Now there are 60848 unique words written strictly in Serbian Latin in the new_list_of_words array
# (Numbers are only for reference and will change if new pdfs are added to the 'books' directory)
# These words will be used as a dataset for neural networks
print(sorted(new_list_of_words))

# Now that we have our dataset we can create vector using word2vec.
# This vector will be used as input to our convolutional autoencoder

# Create CBOW model
model1 = gensim.models.Word2Vec(new_list_of_words, min_count=1)
# print(model1.wv['arsen'])

# Create Skip Gram model
# model2 = gensim.models.Word2Vec(new_list_of_words, min_count=1, vector_size=len(new_list_of_words), window=5, sg=1)

print(model1.wv.similarity('arsen', 'arsenal'))
# print(model2.wv.similarity('arsen', 'arsenal'))
