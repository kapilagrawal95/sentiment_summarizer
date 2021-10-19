import nltk
import sys

from collections import Counter
from nltk.corpus import stopwords

# from external.my_potts_tokenizer import MyPottsTokenizer
import unicodedata

import pickle
import json

import string
import numpy as np
from utils import tokenize, detokenize, createDatasetmaxent
from modelling import get_scores_for_sentences
import pandas as pd

from extract_aspects import sentiment_extractor, extract_aspects, scoreAspects, adjacentsort, get_sentences_by_aspect
from utils import detokenize, tokenize

def summarizer(filename):

	pickle_in = open("sentiment_dict.pickle", "rb")
	sentiment_dict = pickle.load(pickle_in)
	# filename = "/home/themountaindog/Music/project/Amazon_Instant_Video_7.json"
	json1 = open(filename)
	data = json.load(json1)
	reviews = []
	for i in range(len(data)):
		reviews.append(data[i]["reviewText"])
	print("Extracted Reviews...")

	Reviews = []

	for review in reviews:
		review = unicodedata.normalize('NFKD', review).encode('ascii','ignore')
		Reviews.append(review)

	print("Getting sentiments for each sentence in dictionary")
	sent_dictionary, P = sentiment_extractor(Reviews)
	print("Classified Reviews and stored in sent_dictionary")
	print("Extracting aspects...")
	aspects = extract_aspects(Reviews)
	print("Done.")


	###Get Score for Every Aspect
	score_aspects = scoreAspects(aspects, Reviews)

	aspects = adjacentsort(score_aspects, aspects)
	# print "==========="
	print("Aspects:")
	for i,aspect in enumerate(aspects):
		print(str(i) + ". " + aspect)

	L = 4

	aspect = aspects[0]
	t = get_sentences_by_aspect(aspect, Reviews)
	aspect_sentences = []
	for sentence in t:
		aspect_sentences.append(detokenize(sentence))

	for aspect in aspects:
		print(aspect)

		aspect_sentences = get_sentences_by_aspect(aspect, Reviews)
		if(len(aspect_sentences) <= L):
			for sentence in aspect_sentences:
				print("("+str(sent_dictionary[sentence])+")"+ " "+(detokenize(sentence)))
		else:
			pos = 0
			neg = 0
			for sentence in aspect_sentences:
				if sent_dictionary[detokenize(sentence)] == 1:
					pos = pos+1
				elif sent_dictionary[detokenize(sentence)] == -1:
					neg = neg+1
			if(pos+neg == 0):
				for i in range(L):
					print("("+str(sent_dictionary[aspect_sentences[i]])+")"+ " "+(aspect_sentences[i]))
			else:
				ele = [-1, 1]
				prob = [pos/float(pos+neg), neg/float(pos+neg)]
				D = np.random.choice(ele, L, prob)
				visited = [0]*len(aspect_sentences)
				count = 0
				for d in D:
					mx = 0
					# maxi = detokenize(aspect_sentences[0])
					
					for i in range(len(aspect_sentences)):
						results = P[detokenize(aspect_sentences[i])]
						results = map(float, results)
						results = np.array(results)
						k = np.amax(results)
						if d == -1 and sent_dictionary[detokenize(aspect_sentences[i])] == -1:
							k = -k
						if k*d >= mx and visited[i] == 0:
							maxi = detokenize(aspect_sentences[i])
							s = i
					
					print("("+str(sent_dictionary[maxi])+")"+ " "+maxi)
					visited[s] = 1
					count = count+1
							

		print('\n')

if __name__ ==  "__main__":
    summarizer(filename)