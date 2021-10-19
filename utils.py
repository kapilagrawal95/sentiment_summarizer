# from external.my_potts_tokenizer import MyPottsTokenizer
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize

def tokenize(sentence):
	# pt = MyPottsTokenizer(preserve_case=False)
	tokens = word_tokenize(sentence)
	return tokens

def detokenize(sentence):
	sent = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sentence]).strip()
	return sent


def createDatasetmaxent(dat):
	#INPUT: A pandas dataframe which has columns Sentiment, Sentence, Name
	names = dat["Name"].unique()

	# For raw_score append i-1 value and i+1 values
	k = []
	for i in range(len(names)):
		temp = dat[dat["Name"] == names[i]]
		temp = temp["raw_scores"]
		temp = np.array(temp)
		for i in range(len(temp)):
			a = []
			if (i-1<0):
				a.append(0.0)
			else:
				a.append(temp[i-1])
			a.append(temp[i])
			if(i+1>len(temp)-1):
				a.append(0.0)
			else:
				a.append(temp[i+1])
			k.append(a)	
	#For purity_score append i-1 values and i+1 values
	m = []
	for i in range(len(names)):
		temp = dat[dat["Name"] == names[i]]
		temp = temp["purity_scores"]
		temp = np.array(temp)
		for i in range(len(temp)):
			a = []
			if (i-1<0):
				a.append(0.0)
			else:
				a.append(temp[i-1])
			a.append(temp[i])
			if(i+1>len(temp)-1):
				a.append(0.0)
			else:
				a.append(temp[i+1])
			m.append(a)	
	#Append reviews raw_score and purity_score to the dataset		
	s = []
	x = []
	for i in range(len(names)):
		temp = dat[dat["Name"] == names[i]]
		raw_score = 0
		purity = 0
		temp_purity = temp["purity_scores"]
		temp_purity = np.array(temp_purity)
		temp_raw_scores = temp["raw_scores"]
		temp_raw_scores = np.array(temp_raw_scores)
		for j in range(len(temp_raw_scores)):
			raw_score = raw_score + temp_raw_scores[j]
			purity = purity + temp_purity[j]
		for t in range(len(temp)):
			s.append(raw_score)
			x.append(purity)

	#Form dataset with all 8 variables
	features = []

	for i in range(len(k)):
		g = []
		for j in range(len(k[0])):
			g.append(k[i][j])
		for j in range(len(m[0])):
			g.append(m[i][j])
		g.append(s[i])
		g.append(x[i])
		features.append(g)

	return features
