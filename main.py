# importing required modules
import os
import fnmatch
import fitz
import re
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import random
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
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
word2vec_model = gensim.models.Word2Vec(tokenized_words, min_count=1)
# Create Skip Gram model
# word2vec_model = gensim.models.Word2Vec(tokenized_words, min_count=1)

print(word2vec_model.wv['arsen'])
print(word2vec_model.wv.similarity('arsen', 'arsenal'))
# print(word2vec_model.wv.similarity('aritmija', 'aritmičan'))


# Creation of convolutional autoencoder
# Define input shape
input_shape = (None, word2vec_model.vector_size)

# Define encoder layers
input_layer = Input(shape=input_shape)
encoder_layer_1 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(input_layer)
encoder_layer_2 = MaxPooling1D(pool_size=2, padding='same')(encoder_layer_1)
encoder_layer_3 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(encoder_layer_2)
encoder_layer_4 = MaxPooling1D(pool_size=2, padding='same')(encoder_layer_3)

# Define decoder layers
decoder_layer_1 = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(encoder_layer_4)
decoder_layer_2 = UpSampling1D(size=2)(decoder_layer_1)
decoder_layer_3 = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(decoder_layer_2)
decoder_layer_4 = UpSampling1D(size=2)(decoder_layer_3)
output_layer = Conv1D(filters=word2vec_model.vector_size, kernel_size=3, activation='sigmoid', padding='same')(decoder_layer_4)

# Define autoencoder model
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Compile autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Generate training set
x_train = []
for word in random.sample(new_list_of_words, word2vec_model.vector_size):
    x_train.append(word.lower())
# x_train = random.sample(tokenized_words, word2vec_model.vector_size)

# Train the autoencoder model
autoencoder.fit(x_train, x_train, epochs=10, batch_size=32)

# Use the encoder part of the autoencoder to encode some input data
encoder = Model(inputs=input_layer, outputs=encoder_layer_4)
encoded_data = encoder.predict(x_train)

# Use the decoder part of the autoencoder to decode some encoded data
decoder_input_layer = Input(shape=(None, 8))
decoder_layer_1 = autoencoder.layers[-3](decoder_input_layer)
decoder_layer_2 = autoencoder.layers[-2](decoder_layer_1)
decoder_layer_3 = autoencoder.layers[-1](decoder_layer_2)
decoder = Model(inputs=decoder_input_layer, outputs=decoder_layer_3)
decoded_data = decoder.predict(encoded_data)
