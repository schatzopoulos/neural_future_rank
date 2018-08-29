import helpers
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

max_authorlist_length = 20
embedding_size = 8

encoder = LabelEncoder()

def encode_labels(labels):
	encoder.fit(labels)
	return encoder.transform(labels)

def decode_labels(labels):
	return encoder.inverse_transform(labels)

def convert_to_sequences(authors):
	tokenizer = Tokenizer(split=",", lower=True)
	tokenizer.fit_on_texts(authors)
	X = tokenizer.texts_to_sequences(authors)
	X =  pad_sequences(X, maxlen=max_authorlist_length, padding="post", truncating="post")
	return (X, len(tokenizer.word_index) + 1)

log = helpers.get_logger(to_file=False)
input_file = "./datasets/sample_input.csv"

if (len(sys.argv) > 1 and sys.argv[1] == "experiment"):
	input_file = "./datasets/pmc_optimal_cleaned.csv"

log.info("Reading data from file")
authors = helpers.read_field(input_file, "authors")
labels = helpers.read_field(input_file, "5category")

log.info("Converting features to integer sequences")
(X, vocab_size) = convert_to_sequences(authors)
# Y = encode_labels(labels)
Y = to_categorical(labels)

log.info("Building the model")
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, mask_zero=False, input_length=max_authorlist_length))
model.add(Flatten())
model.add(Dense(6, activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

log.info("Fitting the model")
model.fit(X, Y, epochs=50, verbose=0)
loss, accuracy = model.evaluate(X, Y, verbose=0)
print('Accuracy: %f' % (accuracy*100))
print(loss)




