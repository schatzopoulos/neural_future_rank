import helpers
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.utils import class_weight

_max_authorlist_length = 20
_embedding_size = 8
_epochs = 100
_verbose = True

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
	X =  pad_sequences(X, maxlen=_max_authorlist_length, padding="post", truncating="post")
	return (X, len(tokenizer.word_index) + 1)

def define_model(vocab_size, embedding_size, max_authorlist_length):
	model = Sequential()
	model.add(Embedding(vocab_size, embedding_size, mask_zero=False, input_length=max_authorlist_length))
	model.add(Flatten())
	model.add(Dense(6, activation='sigmoid'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
	return model

def compute_weights(y):
	unique_categories = np.unique(y)
	class_weights = class_weight.compute_class_weight('balanced', unique_categories, y)
	
	# transform array to dictionary
	d = dict()
	for i, val in enumerate(unique_categories):
		d[val] = class_weights[i]

	return d

def write_to_file(model, filename="model.json"):
	model_json = model.to_json()
	with open(filename, "w") as json_file:
	    json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights("model.h5")

def load_from_file(filename="model.json"):
	kson_file = open(filename, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")

log = helpers.get_logger(to_file=False)
input_file = "./datasets/sample_input.csv"

if (len(sys.argv) > 1 and sys.argv[1] == "experiment"):
	input_file = "./datasets/pmc_optimal_cleaned.csv"

log.info("Reading data from file")
authors = helpers.read_field(input_file, "authors")
labels = helpers.read_field(input_file, "5category")

log.info("Converting features to integer sequences")
(X, vocab_size) = convert_to_sequences(authors)
Y = to_categorical(labels)

log.info("Building the model")
model = define_model(vocab_size, _embedding_size, _max_authorlist_length)

class_weights = compute_weights(labels)

if (_verbose == True):
	model.summary()
	log.info("Class weights:")
	log.info(class_weights)

log.info("Fitting the model")
model.fit(X, Y, epochs=_epochs, verbose=_verbose)

write_to_file(model)

log.info("Evaluating the model")
loss, accuracy = model.evaluate(X, Y, verbose=_verbose)

log.info('Accuracy: %f' % (accuracy*100))
log.info('Loss: %f' % loss)




