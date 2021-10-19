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

pickle_in = open("sentiment_dict.pickle", "rb")
sentiment_dict = pickle.load(pickle_in)

def sentiment_extractor(reviews):
	sentences = []
	names = []
	i = 0
	for review in reviews:
		for sentence in get_sentences(review):
			sentences.append(sentence)
			names.append(i)
		i = i+1

	raw_scores, purity_scores = get_scores_for_sentences(sentences)
	df = pd.DataFrame(names, columns=['Name'])
	df['raw_scores'] = raw_scores
	df['purity_scores'] = purity_scores

	features = createDatasetmaxent(df)

	tokenized_sentences = []
	for sentence in sentences:
		tokenized_sentences.append(tokenize(sentence))


	df = pd.DataFrame(features,columns=['score_i-1', 'score_i', 'score_i+1', 'purity_i-1', 'purity_i', 'purity_i+1', 'review_score', 'review_purity'])

	filename = "new_dat.csv"
	df.to_csv(filename, sep = ',', index = False)
	x = df.iloc[:, 0:8]
	x = np.array(x)
	##Test Data for Labelling Positive; Negative and Neutral Sentiments
	# load the model from disk
	filename = "finalized_model3.sav"
	clf = pickle.load(open(filename, 'rb'))
	predictions = clf.predict(x)
	scores = clf.predict_proba(x)
	sentences = []
	for sentence in tokenized_sentences:
		sentences.append(detokenize(sentence))

	labels = dict(zip(sentences, predictions))
	P = dict(zip(sentences, scores))
	return labels, P

def extract_aspects(reviews):
	"""
	INPUT: iterable of strings (pd Series, list)
	OUTPUT: list of aspects
	
	Return the aspects from the set of reviews
	"""
	# put all the sentences in all reviews in one stream
	#sentences = []
	#for review in reviews: 
	#	sentences.extend(get_sentences(review))

	tokenized_sentences = []
	for review in reviews:
		for sentence in get_sentences(review):
			tokenized_sentences.append(tokenize(sentence))
			
	# pos tag each sentence
	tagged_sentences = [pos_tag(sentence) for sentence in tokenized_sentences]
	
	# from the pos tagged sentences, get a list of aspects
	aspects = aspects_from_tagged_sents(tagged_sentences)

	return aspects

def get_sentences(review):
	"""
	INPUT: full text of a review
	OUTPUT: a list of sentences

	Given the text of a review, return a list of sentences. 
	"""
	
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	
	if isinstance(review, str):
		return sent_detector.tokenize(review)
	else: 
		raise TypeError('Sentence tokenizer got type %s, expected string' % type(review))

def pos_tag(toked_sentence):
	"""
	INPUT: list of strings
	OUTPUT: list of tuples

	Given a tokenized sentence, return 
	a list of tuples of form (token, POS)
	where POS is the part of speech of token
	"""
	return nltk.pos_tag(toked_sentence)


def aspects_from_tagged_sents(tagged_sentences):
	"""
	INPUT: list of lists of strings
	OUTPUT: list of aspects

	Given a list of tokenized and pos_tagged sentences from reviews
	about a given restaurant, return the most common aspects
	"""

	STOPWORDS = stopwords.words('english') + list(string.punctuation)
	# print STOPWORDS
	# find the most common nouns in the sentences
	noun_counter = Counter()

	for sent in tagged_sentences:
		for word, pos in sent: 
			if pos=='NNP' or pos=='NN' and word not in STOPWORDS:
				# if word != '"':
				noun_counter[word] += 1

	# list of tuples of form (noun, count)
	return [noun for noun, _ in noun_counter.most_common(10)]


def get_sentences_by_aspect(aspect, reviews):
	"""
	INPUT: string (aspect), iterable of strings (full reviews)
	OUTPUT: iterable of strings

	Given an aspect and a list of reviews, return a list 
	sof all sentences that mention that aspect.  
	"""

	tokenized_sentences = []
	for review in reviews:
		for sentence in get_sentences(review):
			tokenized_sentences.append(tokenize(sentence))
	return [sent for sent in tokenized_sentences if aspect in sent]

def get_sentences_general_comments(aspects, reviews):
	tokenized_sentences = []
	for review in reviews:
		for sentence in get_sentences(review):
			tokenized_sentences.append(tokenize(sentence))
	sentences = []
	
	for sent in tokenized_sentences:
		flag = 0
		for aspect in aspects:
			if aspect in sent:
				sentences.append(sent)
				flag = flag + 1
			if (flag == 1):
				break
	return sentences

def rawScore(sent):
	# sent = tokenize(sentence)
	score = 0
	flag = 0
	for word in sent:
		if word == 'not':
			flag = 1
		if word in sentiment_dict:
			k = 1
			if (flag == 1):
				k = -1
				flag = 0
			score = score + k*sentiment_dict[word]

	return score

def purityScore(sentence):
	raw_score = rawScore(sentence)
	length = len(sentence)
	return raw_score/length


def get_scores_for_sentence_in_aspect(aspect, Reviews):
	sentences = get_sentences_by_aspect(aspect, Reviews)
	score_of_sentences = []
	for sentence in sentences:
		score_of_sentences.append(rawScore(sentence))
	return score_of_sentences

def adjacentsort(score, label):
	for i in range(len(score)):
		for j in range(i, len(score)-1):
			if(score[j+1]>score[j]):
				score[j],score[j+1] = score[j+1], score[j]
				label[j],label[j+1] = label[j+1], label[j]

	return label

def scoreAspects(aspects, Reviews):
	score_aspects = []

	for aspect in aspects:
		score = 0
		for sentence in (get_sentences_by_aspect(aspect, Reviews)):
			score = score + rawScore(sentence)
		score_aspects.append(score)
	return score_aspects