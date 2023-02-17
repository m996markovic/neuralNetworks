# importing required modules
import os
import fnmatch
import fitz
import re

import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

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
# Removing all duplicates from the list
words = list(dict.fromkeys(words))

# Creating regex for matching only words written in Serbian Latin alphabet and removing those that do not match it
serbian_latin_alphabet = "abcdefghijklmnopqrstuvwxyzčćđšž"  # define the Serbian Latin alphabet
regex = f'^[{serbian_latin_alphabet}{serbian_latin_alphabet.upper()}]+$'
new_list_of_words = [i for i in words if re.match(regex, i)]
# Now there are n unique words written strictly in Serbian Latin in the new_list_of_words array
# These words will be used as a dataset for neural networks

# Tokenizing words for word2vec algorithm
tokenized_words = [word_tokenize(document.lower()) for document in new_list_of_words]
# Now that we have our dataset we can create vector using word2vec.
# This vector will be used as input to our convolutional autoencoder

# Create CBOW model
model1 = gensim.models.Word2Vec(tokenized_words, min_count=1)
# Create Skip Gram model
model2 = gensim.models.Word2Vec(tokenized_words, min_count=1)

print(model1.wv['arsen'])
print(model1.wv.similarity('arsen', 'arsenal'))
print(model2.wv.similarity('aritmija', 'aritmičan'))
