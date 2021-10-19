import pickle
from external.my_potts_tokenizer import MyPottsTokenizer
import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import average_precision_score
pickle_in = open("sentiment_dict.pickle", "rb")
sentiment_dict = pickle.load(pickle_in)

from utils import tokenize, detokenize, createDatasetmaxent

def rawScore(sentence):
	sent = tokenize(sentence)
	score = 0
	for word in sent:
		if word in sentiment_dict:
			score = score + sentiment_dict[word]

	return score

def purityScore(sentence):
	raw_score = rawScore(sentence)
	length = len(sentence)
	return raw_score/length


def get_scores_for_sentences(sentences):
	#INPUT: A array of sentences

	#OUTPUT: Array of raw scores and purity scores

	# sentences.tolist() #Convert in a list
	raw_scores = [] # Initialize raw scores array
	purity_scores = [] #Initialize purity scores array
	for sentence in sentences:
		raw_scores.append(rawScore(sentence))
		purity_scores.append(purityScore(sentence))

	return raw_scores, purity_scores 

def demo_aspect_extraction():



	data = pd.read_csv("testFile2.csv")
	sentences = data["Sentence"]
	ratings = data["Rating"]
	ratings = np.array(ratings)

	raw_scores, purity_scores = get_scores_for_sentences(sentences)

	data['raw_scores'] = raw_scores
	data['purity_scores'] = purity_scores

	#Remove noisyness by removing all the invalid ratings apart from 0 to 5
	data = data.loc[data["Rating"].isin(['0', '1', '2', '3', '4', '5'])]

	#Create a new dataset with raw scores, purity scores, and review name to train max-ent classifier
	dat = data.iloc[:, 3:]
	features = createDatasetmaxent(dat)
	full = []
	for i in range(len(features)):
		g = []
		for j in range(len(features[0])):
			g.append(features[i][j])
		g.append(ratings[i])
		full.append(g)

	#Write a csv of the dataset with both features and labels.	{OPTIONAL}
	df = pd.DataFrame(features, columns=['score_i-1', 'score_i', 'score_i+1', 'purity_i-1', 'purity_i', 'purity_i+1', 'review_score', 'review_purity'])
	# df['ratings'] = ratings
	sentiments = data["Sentiment"]
	sentiments = np.array(sentiments)
	df["Sentiment"] = sentiments
	file_name = "dataset.csv"
	df.to_csv(file_name, sep=',', index = False)

	#Fitting a Max-Entropy Logistic Regression Classifier on the dataset
	x = features
	y = sentiments
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=0)
	# dict1 = {1 : 0.6, 0 : 0.1, -1 : 0.2}
	clf = LogisticRegression()
	clf.fit(x_train, y_train)
	predictions = clf.predict(x_test)
	pred_scores = clf.predict_proba(x_test)
	pred_scores = np.array(pred_scores)
	pred_pos = pred_scores[:, 2]
	pred_neg = pred_scores[:, 0]
	#Print accuracy and confusion matrix for the classifier.
	score = clf.score(x_test, y_test)
	print(score)
	cm = metrics.confusion_matrix(y_test, predictions)
	print(cm)
	y_pos = []
	for y in y_test:
		if y == 0 or y == -1:
			y_pos.append(0)
		else:
			y_pos.append(1)

	y_neg = []
	for y in y_test:
		if y == 0 or y == 1:
			y_neg.append(0)
		else:
			y_neg.append(1)

	pos_avg = average_precision_score(y_pos, pred_pos)
	neg_avg = average_precision_score(y_neg, pred_neg)
	print "Positive Average Precision" + str(pos_avg)
	print "Negative Average Precision" + str(neg_avg)
	# save the model to disk
	filename = 'finalized_model.sav'
	pickle.dump(clf, open(filename, 'wb'))

if __name__ == "__main__":
	demo_aspect_extraction()