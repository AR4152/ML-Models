import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ['I love my dog', 'I love my cat', 'Cat', 'You love my dog!', 'Do you think my dog is amazing?']

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)

print(word_index)
print(sequences)
print(padded)

# test_sentences = ['Hi! I love learning TensorFlow', 'This is amazing']
# test_sequences = tokenizer.texts_to_sequences(test_sentences)
# print(test_sequences)
