import re

from jenks import jenks
import numpy as np
def goodness_of_variance_fit(array, classes):
	# get the break points
	classes = jenks(array, classes)
	print classes
	# do the actual classification
	classified = np.array([classify(i, classes) for i in array])

	# max value of zones
	maxz = max(classified)

	# nested list of zone indices
	zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]

	# sum of squared deviations from array mean
	sdam = np.sum((array - array.mean()) ** 2)

	# sorted polygon stats
	array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]

	# sum of squared deviations of class means
	sdcm = sum([np.sum((classified - classified.mean()) ** 2) for classified in array_sort])

	# goodness of variance fit
	gvf = (sdam - sdcm) / sdam

	return gvf

def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1


def split(str):
	return re.split(r'\t+', str)

filename = "./pmc_optimal_fcc_2013_2016.txt"

scores = []
fd = open(filename) 
for line in  fd.readlines():
	score = int(split(line)[1].rstrip())
	scores.append(score)

scores = np.array(scores)
gvf = 0.0
nclasses = 4
while gvf < .8:
	print "k: " + str(nclasses)
	gvf = goodness_of_variance_fit(scores, nclasses)
	print str(gvf)
	nclasses += 1
	if nclasses > 6:
		break

