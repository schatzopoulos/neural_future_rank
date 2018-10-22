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
from sklearn.model_selection import StratifiedKFold

_max_authorlist_length = 20
_embedding_size = 8
_epochs = 50
_verbose = False
_seed = 7

np.random.seed(_seed)

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
	model.add(Dense(6, activation='softmax'))
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
Y = helpers.read_field(input_file, "5category")

# compute weights for each class / imbalanced data
class_weights = compute_weights(Y)

log.info("Converting features to integer sequences")
(X, vocab_size) = convert_to_sequences(authors)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=_seed)
cvscores = []

epoch = 0
for train, test in kfold.split(X, Y):
	print("Epoch", epoch)
	epoch += 1

  	# Create model
	log.info("Building the model")
	model = define_model(vocab_size, _embedding_size, _max_authorlist_length)

	# Compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

	# Fit the model
	if (_verbose == True):
		model.summary()
		log.info("Class weights:")
		log.info(class_weights)

	log.info("Fitting the model")
	model.fit(X[train], to_categorical(Y[train]), epochs=_epochs, verbose=_verbose, class_weight=class_weights)

	# evaluate the model
	log.info("Evaluating the model")
	scores = model.evaluate(X[test], to_categorical(Y[test]), verbose=_verbose)

	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
	print()

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
