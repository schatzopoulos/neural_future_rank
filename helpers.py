import logging
import pandas as pd 

def get_logger():
	# create logger with 'spam_application'
	log = logging.getLogger('neural app')
	log.setLevel(logging.DEBUG)

	# create formatter and add it to the handlers
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

	# create file handler which logs even debug messages
	fh = logging.StreamHandler()

	# uncomment below to redirect logs to file
	# fh = logging.FileHandler('spam.log')

	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)
	log.addHandler(fh)
	return log


def read_field(file, field_name): 
	df = pd.read_csv(file, sep='\t', usecols=[field_name])
	return df.to_records(index=False)[field_name]