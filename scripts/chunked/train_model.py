import tensorflow
import keras
from keras import layers
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy
import os
import os.path
import gc
import json

####################################################################################################

def Flatten(sentence):
	out = ""
	for t in sentence:
		out += t
	return out

def Atomize(tokens):
	new_tokens = []
	for s in tokens:
		inner = []
		sentence = Flatten(s)
		for c in sentence:
			inner.append(c)
		new_tokens.append(inner)
	return new_tokens

# Takes a string and applies the rules to produce byte pairs.
def ApplyRules(sentence, merge_rules):
	
	sentence = list(sentence)

	for rule in merge_rules:
		new_sentence = []
		search_pair = rule[0] + rule[1]

		i = 0
		while i < len(sentence):
			c0 = sentence[i]

			if i < len(sentence) - 1:
				c1 = sentence[i + 1]
					
				if c0 + c1 == search_pair:
					new_sentence.append(search_pair)
					i += 2
					continue
			
			new_sentence.append(c0)
			i += 1

		sentence = new_sentence
		
	return sentence

####################################################################################################

directory_name = "Cleaned_ACTib_Tiny"
corpus_name = "cleaned_actib_corpus_tiny"
corpus_file_num = 20

vec_size = 400

net_major_batch_size = 5000
net_minor_batch_size = 64
net_epochs = 3
net_window_size = 4
net_hidden_layer_size = 500
net_hidden_layer_count = 5
net_learning_rate = 1e-5

test_frequency = 20

####################################################################################################

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

tokens = []

print("Loading dataset.")
for i in range(1, corpus_file_num + 1):
	with open(os.path.dirname(__file__) + "/../../output/" + directory_name + "/Tokens/" + corpus_name + "_" + str(i) + "-tokens.json", "r", encoding = 'utf-8') as f:
		print('\r' + str(i) + " / " + str(corpus_file_num), end = '')
		temp_tokens = json.load(f)
		for s in temp_tokens:
			tokens.append(s)

print()

print("Loading merge rules.")
with open(os.path.dirname(__file__) + "/../../output/" + directory_name + "/Merge_Rules/" + corpus_name + "-merge_rules.json", "r", encoding = 'utf-8') as f:
	merge_rules = json.load(f)

####################################################################################################

print("Loading embeddings.")
vectors = KeyedVectors.load(os.path.dirname(__file__) + "/../../output/" + directory_name + "/Embeddings/bod_bp.vectors", mmap='r')

print("Setting up model.")

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

# Now, we need to construct our train and test datasets.
# This involves running a window over the dataset, collecting
# n characters around each target character, running the merge rules
# over the window and then getting the vectors of the resulting tokens.

print("Preprocessing data.")

iterations = 0
sentence_start = 0

while True:

	gc.collect()

	x_data = []
	y_data = []

	sentence_end = min(sentence_start + net_major_batch_size, len(tokens) - 1)

	print("Section: " + str(sentence_start) + " -> " + str(sentence_end) + " / " + str(len(tokens) - 1))

	for sentence in tokens[sentence_start:sentence_end]:
		flat_sentence = list(Flatten(sentence))
		i = 0
		for word in sentence:
			j = 0
			for character in word:
				if i + 1 == len(flat_sentence) - 1:
					continue

				start = int(max(0, i - (net_window_size / 2)))
				end = int(min(len(flat_sentence) - 1, i + 1 + (net_window_size / 2)))

				section = flat_sentence[start:end]

				byte_pairs = ApplyRules(section, merge_rules)

				embeddings = []

				for bp in byte_pairs:
					if bp in vectors.key_to_index:
						e = vectors[bp]
					else:
						e = [0.0] * vec_size
					embeddings.extend(e)

				l = (net_window_size + 1) - len(byte_pairs)

				if l > 0:
					e = [0.0] * vec_size * l
					embeddings.extend(e)

				y = 0.0

				if j == len(word) - 1:
					y = 1.0
				
				x_data.extend(embeddings)
				y_data.append(y)

				i += 1
				j += 1

	x_data = numpy.array(x_data)
	y_data = numpy.array(y_data)

	x_data = x_data.reshape(len(y_data), (net_window_size + 1) * vec_size)
	sentence_start = sentence_end

	iterations += 1
	
	if iterations % 20 == 0:
		results = seg_model.evaluate(x_data, y_data, batch_size=64)
	else:
		history = seg_model.fit(
			x_data,
			y_data,
			batch_size=net_minor_batch_size,
			epochs=net_epochs,
			verbose=1
		)

	if sentence_end == len(tokens) - 1:
		break

print("Saving model.")

seg_model.save(os.path.dirname(__file__) + "/../../output/" + directory_name + "/Model/bod_bp_seg.keras")

print("Job done.")

####################################################################################################