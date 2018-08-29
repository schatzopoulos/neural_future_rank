import logging
import pandas as pd 

def get_logger(to_file=False):
	# create logger with 'spam_application'
	log = logging.getLogger('neural app')
	log.setLevel(logging.DEBUG)

	# create formatter and add it to the handlers
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

	# create file handler which logs even debug messages
	if (to_file == False):
		fh = logging.StreamHandler()
	else: 
		fh = logging.FileHandler('regression.log')

	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)
	log.addHandler(fh)
	return log


def read_field(file, field_name): 
	df = pd.read_csv(file, sep='\t', usecols=[field_name])
	return df.to_records(index=False)[field_name]