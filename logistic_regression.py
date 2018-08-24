import helpers
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

log = helpers.get_logger()

input_file = "./datasets/sample_input.csv"

if (len(sys.argv) > 1 and sys.argv[1] == "experiment"):
	input_file = "./datasets/pmc_optimal_cleaned.csv"

log.info("Reading data from file")
authors = helpers.read_field(input_file, "authors")
labels = helpers.read_field(input_file, "5category")

log.info("Converting features to integer sequences")
tokenizer = Tokenizer(split=",", lower=True)
tokenizer.fit_on_texts(authors)
X = tokenizer.texts_to_sequences(authors)
X = pad_sequences(X, maxlen=15, padding="post", truncating="post")
# print(X)



