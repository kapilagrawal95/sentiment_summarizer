from Tkinter import Tk

import Tkinter, Tkconstants, tkFileDialog
import nltk
import sys

from collections import Counter
from nltk.corpus import stopwords

import unicodedata

import pickle
import json

import string
import numpy as np
from utils import tokenize, detokenize, createDatasetmaxent
from modelling import get_scores_for_sentences
import pandas as pd

import math

from extract_aspects import sentiment_extractor, extract_aspects, scoreAspects, adjacentsort, get_sentences_by_aspect, get_sentences_general_comments
from utils import detokenize, tokenize

def summarizer(reviews):

	pickle_in = open("sentiment_dict.pickle", "rb")
	sentiment_dict = pickle.load(pickle_in)
	

	Reviews = []

	for review in reviews:
		review = unicodedata.normalize('NFKD', review).encode('ascii','ignore')
		Reviews.append(review)

	# print("Getting sentiments for each sentence in dictionary")
	sent_dictionary, P = sentiment_extractor(Reviews)
	# print("Classified Reviews and stored in sent_dictionary")
	# print("Extracting aspects...")
	aspects = extract_aspects(Reviews)
	# print("Done.")
	main_aspects = []
	### Taking 10 aspects
	for i in range(3):
		main_aspects.append(aspects[i])

	###Get Score for Every Aspect
	score_aspects = scoreAspects(aspects, Reviews)

	aspects = adjacentsort(score_aspects, aspects)
	main_aspects = []
	### Taking 10 aspects

	aspects_to_show = 8
	for i in range(aspects_to_show-1):
		main_aspects.append(aspects[i])
	other_aspects = []
	i = i+1
	while i<10:
		other_aspects.append(aspects[i])
		i = i+1


	main_aspects.append('general comments')

	# print(other_aspects)
	
	# print("Aspects:")
	# for i,aspect in enumerate(main_aspects):
	# 	print(str(i) + ". " + aspect)

	L = 4

	for aspect in main_aspects:
		if aspect == 'general comments':
			aspect_sentences = get_sentences_general_comments(other_aspects, Reviews)
		else:
			aspect_sentences = get_sentences_by_aspect(aspect, Reviews)
		pos = 0
		neg = 0
		for sentence in aspect_sentences:
			if sent_dictionary[detokenize(sentence)] == 1:
				pos = pos+1
			elif sent_dictionary[detokenize(sentence)] == -1:
				neg = neg+1
		comments = len(aspect_sentences)
		stars = math.ceil((pos*5)/float(comments))
		print str(aspect) + " ("+str(comments)+" Comments) " + str(stars) + "/5 stars"
		print "Positive: " + str(pos) +"  Negative: " + str(neg)
		if(len(aspect_sentences) <= L):
			for sentence in aspect_sentences:
				print("("+str(sent_dictionary[detokenize(sentence)])+")"+ " "+(detokenize(sentence)))
		else:
			if(pos+neg == 0):
				length = L
				for sentence in aspect_sentences:
					if(length == 0):
						break
					print "("+str(sent_dictionary[detokenize(sentence)])+")"+ " "+detokenize(sentence)
					length = length-1
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

def main2():
	Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
	filename = tkFileDialog.askopenfilename() # show an "Open" dialog box and return the path to the selected file
	json1 = open(filename)
	data = json.load(json1)
	reviews = []
	for i in range(len(data)):
		reviews.append(data[i]["reviewText"])
	print("Extracted Reviews...")
	summarizer(reviews)

if __name__ ==  "__main__":
    main2()