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

vec_min_count = 5
vec_window = 8
vec_size = 400
vec_sample = 6e-5
vec_alpha = 0.03
vec_min_alpha = 0.0007
vec_negative = 20
vec_workers = 4
vec_epochs = 30

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

tokens_path = "cleaned_actib_corpus_test-tokens"
byte_pairs_path = "cleaned_actib_corpus_test-byte_pairs"

with open(os.path.dirname(__file__) + "/../datasets/" + tokens_path + ".json", "r", encoding = 'utf-8') as f:
	tokens = json.load(f)

with open(os.path.dirname(__file__) + "/../datasets/" + byte_pairs_path + ".json", "r", encoding = 'utf-8') as f:
	byte_pairs = json.load(f)

test_train_cutoff = round(len(tokens) * data_train_cutoff)
x_train = tokens[:test_train_cutoff]
x_test = tokens[test_train_cutoff:]

####################################################################################################

# vec_model = Word2Vec(min_count=vec_min_count, window=vec_window, vector_size=vec_size,
# 				 sample=vec_sample, alpha=vec_alpha, min_alpha=vec_min_alpha,
# 				 negative=vec_negative, workers=vec_workers)
# vec_model.build_vocab(byte_pairs, progress_per=10000)
# vec_model.train(byte_pairs, total_examples=vec_model.corpus_count, epochs=vec_epochs, report_delay=1)

# vec_model.save("bod_vector_test.model")

# sylvecs = vec_model.wv
# sylvecs.save("bod_test.vectors")
# sylvecs.save_word2vec_format("bod_vectors_test.txt", binary=False, write_header=False)

# ####################################################################################################

# inputs = keras.Input(shape=(vec_size * (net_window_size + 1),), name="digits")
# x = layers.Dense(net_hidden_layer_size, activation="relu", name="dense_1")(inputs)
# for i in range(2, net_hidden_layer_count):
# 	x = layers.Dense(net_hidden_layer_size, activation="relu", name="dense_" + str(i))(x)
# outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

# seg_model = keras.Model(inputs=inputs, outputs=outputs)

# seg_model.compile(
#     optimizer=keras.optimizers.RMSprop(learning_rate=net_learning_rate),
#     loss=keras.losses.BinaryCrossentropy(from_logits=False),
#     metrics=[keras.metrics.BinaryAccuracy(),
# 			 keras.metrics.F1Score(),
# 			 keras.metrics.Precision(),
# 			 keras.metrics.Recall()]
# )

# ####################################################################################################

# index = 0

# while index < len(tokens) - 1:
# 	batch_size = min(net_major_batch_size, len(tokens) - index - 1)
	
# 	x_vec_data = []
# 	y_vec_data = []
	
# 	for i in range(index, index + batch_size):
# 		sentence = tokens[i]
		
		

# 	index += batch_size

# 	if index >= len(tokens) - 1:
# 		break

# x_test = []
# y_test = []

####################################################################################################