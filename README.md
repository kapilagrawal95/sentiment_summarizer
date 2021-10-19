# sentiment_summarizer


----------------------------------------------------------------UPDATE-0------------------------------------------------------------------------

Present:

1. A maximum entropy classifier that predicts each sentence of a review whether it is positive or negativee.

2. A dynamic extractor that uses syntactic patterns in reviews to find aspects and score them.

3. A summarizer that displays n aspects with L sentences each (if present) using a heuristic approach as discussed in the paper.


Resources:

1. A sentiment labelled file names testFile2.csv for training a classifier into finalized_model.sav

2. A lexicon obtained from wordnet score propagation sentiment_dict whose algorithm is yet to be published.

3. Extracted Reviews from Amazon Video Dataset in json. (Note: only a small subset of 100 reviews have been extracted because of poor computation power)
Note: some minor modifications are needed to make in the json file for the proper loading.

4. Two csvs namely new_dat.csv and dataset.csv are intermediate files that I created for my reference. You can ignore them for now.


To Do:

1. Some code of Summarizer is added to extract_aspects.py. Need to change and create a seperate file for summarizer

2. Add WordNetScore Propagation Algorithm

3. Some minor tweaks in dynamic extractor for aspects.

4. Make json loading more smooth for loading reviews to be summarized.


----------------------------------------------------------------UPDATE-1------------------------------------------------------------------------

Dependencies:
1. Python2
2. nltk
3. Tkinter
4. pandas

Python Scripts:

1. main.py
	This file contains the implementation of Section 4.
	INPUT: A list of json objects passed as .json. See Amazon_Instant_Video_7.json for references.
	OUTPUT: Prints on the terminal all the extracted aspects with 5 sentences (if present) consisting of opinions and labelled.

2. wordnetprop.py
	This file constructs the Sentiment Lexicon and stores in sentiment_dict3.pickle to be used by modelling and extract_aspects


3. modelling.py
	This file trains a max-entropy classifier for sentiment-laden sentences and stores the classifiers as finalized_model.sav

4. extract_aspects.py
	This file is the implementation of Section 3.1 and 3.3. Mainly this file consists only of functions that are called by gui.py

5. utils.py
	This file contains intermediate functions being called upon by all the files.s
	

To Do:
1. Add General Comments as Aspect in main.py
2. Revisit the code of extract_aspects.py for amends


----------------------------------------------------------------UPDATE-2------------------------------------------------------------------------

Checklist for To-Do from Updates 1:
1. checked

To do:
1. Rewrite extract_aspects.py as well as wordnetprop.py
2. Add new test cases from the amazon folder.
