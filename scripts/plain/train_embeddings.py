import os
import json
from gensim.models import Word2Vec

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

####################################################################################################

directory_name = "Cleaned_ACTib_Full"
corpus_name = "cleaned_actib_corpus"

byte_pairs = []

print("Loading dataset.")
with open(os.path.dirname(__file__) + "/../../output/" + directory_name + "/" + corpus_name + "-byte_pairs.json", "r", encoding = 'utf-8') as f:
	byte_pairs = json.load(f)
print()
print("Preparing model.")

vec_model = Word2Vec(min_count=vec_min_count, window=vec_window, vector_size=vec_size,
					sample=vec_sample, alpha=vec_alpha, min_alpha=vec_min_alpha,
					negative=vec_negative, workers=vec_workers)
vec_model.build_vocab(byte_pairs, progress_per=10000)

print("Training.")
vec_model.train(byte_pairs, total_examples=vec_model.corpus_count, epochs=vec_epochs, report_delay=1)

vec_model.save(os.path.dirname(__file__) + "/../../output/" + directory_name + "/bod_bp_vectors.model")

sylvecs = vec_model.wv
sylvecs.save(os.path.dirname(__file__) + "/../../output/" + directory_name + "/bod_bp.vectors")
sylvecs.save_word2vec_format(os.path.dirname(__file__) + "/../../output/" + directory_name + "/bod_bp_vectors.txt", binary=False, write_header=False)

####################################################################################################