# This is code learned from sentdex's tutorial https://www.youtube.com/user/sentdex

# This is the code that produces an input vector for the neural network, we store in sentiment_set.pickle in
# the code directory. At line 38, we print the size of input vector.

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000

# If we get memory error, we basically ran out of RAM, especially on CPU

def create_lexicon(pos, neg):
	lexicon = []
	for fi in [pos,neg]:
		with open(fi, 'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				# hm_lines is the maximum lines
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)
				# The lexicon at this point contains the copied words

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	# w_counts as of now will be a dictionary
	# {'the': 1232, 'an': 1223}
	l2 = []
	# we don't want super common words so we are going to remove them, here he uses
	# a custom code, to remove all the words and cap them in between 1000 and 50
	# Other mathod is to remove the stop-words, I am still working on that 
	for w in w_counts:
		if 1000 > w_counts[w] > 50:
			l2.append(w)

	print("Vector Size =",len(l2))

	return l2

def sample_handling(sample, lexicon, classification):
	feature_set = []
	'''
	the features_set is a list of lists that is where each index is a list of
	features and its classification
	[
	[[0 1 1 1 1 0], [0, 1]],
	the feature set is [0 1 1 1 1 0] and it's classification is [0, 1]
	[[1,1,3,7,0,0], [1, 0]]
	]
	'''
	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			# creating a empty array of length equal to lexicon
			features = np.zeros(len(lexicon))
			for word in current_words:
				# iterate through every single word
				if word.lower() in lexicon:
					# search for word.lower() in lexicon, if True
					# find the index value of the word.lower()
					index_value = lexicon.index(word.lower())
					# increment the index value by 1
					# this is basically a self made tokenizer of sorts, where
					# the features array is the number of words present in each
					# sentence
					features[index_value] += 1

			features = list(features)
			feature_set.append([features, classification])

	return feature_set

def create_features_set_and_labels(pos, neg, test_size = 0.1):
	# the function that will combine it all
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling('pos.txt', lexicon, [1,0])
	features += sample_handling('neg.txt', lexicon, [1, 0])

	# We don't want the traiing to be in a continous manner, i.e. to learn positive first and then negatives
	# so we shuffle them
	random.shuffle(features)

	features = np.array(features)
	testing_size = int(test_size * len(features))

	# when we want all the 0th elements we use this notation [:, 0]
	# thus train_x is the list of all the features till the last test_size
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return (train_x, train_y, test_x, test_y)

if __name__ == '__main__':
	(train_x, train_y, test_x, test_y) = create_features_set_and_labels('pos.txt', 'neg.txt')
	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)