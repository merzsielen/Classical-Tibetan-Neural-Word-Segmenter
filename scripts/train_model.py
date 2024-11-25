import tensorflow
import keras
from keras import layers
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy
import os
import os.path
import gc
from collections import Counter, defaultdict
from heapq import heappush, heappop
import time

# Takes a text and returns set of sets of the tokens in each sentence.
def Tokenize(text):

	outer = []
	out = []
	syl = ""

	iter = 0
	for i in text:
		iter = iter + 1

		if (i == ' ' and syl == ""):
			continue

		if (i == '\n'):
			if (syl != ''):
				out.append(syl)
			outer.append(out)
			syl = ""
			out = []
			continue

		if (i == ' '):
			out.append(syl)
			syl = ""
			continue

		syl += i
		
	return outer

# Takes a vector of strings and flattens it into a single string.
def Flatten(sentence):
	out = ""
	for t in sentence:
		out += t
	return out

def IsInUnicodeRange(c, start, end):
	if (len(c) > 1): return False
	char_code = ord(c)
	return (start <= char_code <= end)

def BPM_1(corpus, num_merges):

	merge_rules = {}

	for n in range(num_merges):
		frequency_dictionary = {}

		for sentence in corpus:
			
			if (len(sentence) == 0): continue

			c0 = sentence[0]
			for c1 in sentence[1:]:
				pair = (c0, c1)
				
				if (c0[len(c0) - 1] == "་" or c0 == "།" or c1 == "།" or IsInUnicodeRange(c0, 3953, 3969)):
					c0 = c1
					continue

				if pair in frequency_dictionary.keys():
					frequency_dictionary[pair] += 1
				else:
					frequency_dictionary[pair] = 1
				
				c0 = c1
		
		most_frequent_pair = ("", "")
		most_frequent_comb = ""
		max_frequency = 0

		for k in frequency_dictionary.keys():
			if frequency_dictionary[k] > max_frequency:
				most_frequent_pair = k
				most_frequent_comb = k[0] + k[1]
				max_frequency = frequency_dictionary[k]

		merge_rules[most_frequent_pair] = most_frequent_comb

		output = []
		for sentence in corpus:
			
			if (len(sentence) == 0): continue

			inner = []

			skip = False
			c0 = sentence[0]
			for c1 in sentence[1:]:
				pair = c0 + c1

				if skip:
					skip = False
					c0 = c1
					continue

				if pair == most_frequent_comb:
					skip = True
					inner.append(pair)
					c0 = c1
					continue
				
				inner.append(c0)
				c0 = c1

			if not skip:
				inner.append(sentence[len(sentence) - 1])

			output.append(inner)
		corpus = output
	return corpus, merge_rules

def BPM_2(corpus, num_merges):

	pair_freq = Counter()
	for sentence in corpus:
		for j in range(len(sentence) - 1):
			c0 = sentence[j]
			c1 = sentence[j + 1]
			if (c0[len(c0) - 1] == "་" or c0 == "།" or c1 == "།" or IsInUnicodeRange(c0, 3953, 3969)):
				pair_freq[(c0, c1)] += 1

	pq = []
	for pair, freq in pair_freq.items():
		heappush(pq, (-freq, pair))

	merge_rules = {}

	for i in range(num_merges):
		if not pq:
			break

		freq, (first, second) = heappop(pq)
		freq = -freq

		if pair_freq.get((first, second), 0) != freq:
			continue

		new_token = first + second
		merge_rules[(first, second)] = new_token

		new_corpus = []
		for sentence in corpus:
			new_sentence = []
			j = 0
			while j < len(sentence):
				if j < len(sentence) - 1:
					if sentence[j] == first and sentence[j + 1] == second:
						new_sentence.append(new_token)
						j += 2
						continue
				new_sentence.append(sentence[j])
				j += 1
			new_corpus.append(tuple(new_sentence))
		corpus = new_corpus

		pair_freq = Counter()
		for sentence in corpus:
			for j in range(len(sentence) - 1):
				c0 = sentence[j]
				c1 = sentence[j + 1]
				if (c0[len(c0) - 1] == "་" or c0 == "།" or c1 == "།" or IsInUnicodeRange(c0, 3953, 3969)):
					pair_freq[(c0, c1)] += 1
				

		pq = []
		for pair, freq in pair_freq.items():
			heappush(pq, (-freq, pair))

	return corpus, merge_rules

# Takes tokens, pairs them a certain number of iterations, and spits out x_train, y_train, x_test, and y_test.
def PrepareBytePairs(tokens, pair_iterations):

	# First, we separate out each sentence into a vector of characters. Then we iteratively
	# merge these based on frequency.

	paired_tokens = []
	for s in tokens:
		inner = []
		sentence = Flatten(s)
		for c in sentence:
			inner.append(c)
		paired_tokens.append(inner)

	paired_tokens, pair_rules = BPM_1(paired_tokens, pair_iterations)

	return paired_tokens

####################################################################################################

vec_min_count = 5
vec_window = 8
vec_size = 400
vec_sample = 6e-5
vec_alpha = 0.03
vec_min_alpha = 0.0007
vec_negative = 20
vec_workers = 4
vec_epochs = 30
pair_iterations = 500

net_major_batch_size = 10000
net_minor_batch_size = 64
net_major_epochs = 20
net_minor_epochs = 1
net_window_size = 2
net_hidden_layer_size = 500
net_hidden_layer_count = 5
net_learning_rate = 1e-5

data_train_cutoff = 0.8

####################################################################################################

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

file_name = "cleaned_actib_corpus_test"

text = open(os.path.dirname(__file__) + "/../datasets/" + file_name + ".txt", "r", encoding = 'utf-8')
corpus = text.read()
tokens = Tokenize(corpus)
byte_pairs = PrepareBytePairs(tokens, pair_iterations)

test_train_cutoff = round(len(tokens) * data_train_cutoff)
x_train = tokens[:test_train_cutoff]
x_test = tokens[test_train_cutoff:]

####################################################################################################

vec_model = Word2Vec(min_count=vec_min_count, window=vec_window, vector_size=vec_size,
				 sample=vec_sample, alpha=vec_alpha, min_alpha=vec_min_alpha,
				 negative=vec_negative, workers=vec_workers)
vec_model.build_vocab(byte_pairs, progress_per=10000)
vec_model.train(byte_pairs, total_examples=vec_model.corpus_count, epochs=vec_epochs, report_delay=1)

vec_model.save("bod_vector_test.model")

sylvecs = vec_model.wv
sylvecs.save("bod_test.vectors")
sylvecs.save_word2vec_format("bod_vectors_test.txt", binary=False, write_header=False)

####################################################################################################

inputs = keras.Input(shape=(vec_size * (net_window_size + 1),), name="digits")
x = layers.Dense(net_hidden_layer_size, activation="relu", name="dense_1")(inputs)
for i in range(2, net_hidden_layer_count):
	x = layers.Dense(net_hidden_layer_size, activation="relu", name="dense_" + str(i))(x)
outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

seg_model = keras.Model(inputs=inputs, outputs=outputs)

seg_model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=net_learning_rate),
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[keras.metrics.BinaryAccuracy(),
			 keras.metrics.F1Score(),
			 keras.metrics.Precision(),
			 keras.metrics.Recall()]
)

####################################################################################################

index = 0

while index < len(tokens) - 1:
	batch_size = min(net_major_batch_size, len(tokens) - index - 1)
	
	x_vec_data = []
	y_vec_data = []
	
	for i in range(index, index + batch_size):
		sentence = tokens[i]
		
		

	index += batch_size

	if index >= len(tokens) - 1:
		break

x_test = []
y_test = []

####################################################################################################