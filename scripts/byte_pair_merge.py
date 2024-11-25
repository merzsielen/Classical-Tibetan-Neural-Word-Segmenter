import os
import os.path
import json

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

def BPM(corpus, num_merges):

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

# Takes tokens, pairs them a certain number of iterations, and spits out paired tokens.
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

	paired_tokens, pair_rules = BPM(paired_tokens, pair_iterations)

	return paired_tokens

####################################################################################################

file_name = "cleaned_actib_corpus"
pair_iterations = 500

####################################################################################################

text = open(os.path.dirname(__file__) + "/../datasets/" + file_name + ".txt", "r", encoding = 'utf-8')
corpus = text.read()
tokens = Tokenize(corpus)
byte_pairs = PrepareBytePairs(tokens, pair_iterations)

with open(os.path.dirname(__file__) + "/../datasets/" + file_name + "-tokens.json", "w", encoding = 'utf-8') as f:
	json.dump(tokens, f)

with open(os.path.dirname(__file__) + "/../datasets/" + file_name + "-byte_pairs.json", "w", encoding = 'utf-8') as f:
	json.dump(byte_pairs, f)