from collections import defaultdict
import numpy
from nltk.corpus import wordnet as wn
from nltk.corpus import WordNetCorpusReader
from operator import itemgetter
import pickle

def pos(lemma):
    return lemma.synset().pos()

class SentimentLexicon:
    def __init__(self, positive, negative, neutral, pos, start=0, finish=None, weight=0.2, rescale = False):
        self.positive = positive
        self.negative = negative
        self.neutral = neutral
        self.pos = pos
        self.weight = weight
        self.rescale = rescale                                
        
        self.s = {}
        self.s0 = {}
        self.initialize_s()
        
        self.lemmas = sorted(self.s.keys(), cmp=(lambda x, y : cmp(x.name, y.name)))
        self.lemma_count = len(self.lemmas)
        self.start = start
        self.finish = finish
        if self.finish == None or self.finish > self.lemma_count:
            self.finish = self.lemma_count       
        self.a = defaultdict(lambda : defaultdict(float))
        self.initialize_a()
    
    def iterate(self, runs=5):
        for i in range(runs):
            self.multiply_matrices()
        sentiment = {}        
        for lemma,score in self.s.items():
            if self.a[lemma]:
                sentiment[lemma.name()] = self.rescale_score(score)
        return sentiment


    def rescale_score(self, score):
        if self.rescale:
            if abs(score) <= 1:
                return 0.0
                
            else:
                return numpy.log(abs(score)) * numpy.sign(score)
                
        else:
            return score
 

    def initialize_s(self):

        synsets = list(wn.all_synsets(pos=self.pos))
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name() in self.positive:
                    self.s0[lemma] = 1.0
                    self.s[lemma]  = 1.0                   
                elif lemma.name() in self.negative:
                    self.s0[lemma] = -1.0
                    self.s[lemma]  = -1.0                  
                else:
                    self.s0[lemma] = 0.0
                    self.s[lemma]  = 0.0

    def initialize_a(self):
        
        
        for index in range(self.start, self.finish):
                    lemma1 = self.lemmas[index]

                    self.a[lemma1][lemma1] = 1 + self.weight
                    if lemma1.name() not in self.neutral:

                        this_pos = lemma1.synset().pos()
                        if this_pos == "s":
                            this_pos = "a"
                        syns = wn.synsets(lemma1.name(), this_pos)                    

                        for syn in syns:
                            for lemma2 in syn.lemmas():
                                if lemma1 != lemma2:
                                    self.a[lemma1][lemma2] = self.weight
                                    ants = [ant for syn in syns
                                            for lem in syn.lemmas()
                                            for ant in lem.antonyms() if lem == lemma1]
                                    for lemma2 in ants:
                                        self.a[lemma1][lemma2] = -self.weight


    def multiply_matrices(self):
        for lemma1 in self.lemmas:
            if self.a[lemma1]:
                lemma1_vals = self.a[lemma1]
                colsum = sum(self.s[lemma2] * lemma1_vals[lemma2]
                             for lemma2 in self.lemmas if lemma1_vals[lemma2] != 0.0 and self.s[lemma2] != 0.0)
                self.s[lemma1] = self.sign_correct(lemma1, colsum)

    def sign_correct(self, lemma, colsum):

        if numpy.sign(self.s0[lemma]) != numpy.sign(colsum):
            return -colsum
        else:
            return colsum

def tiny_adv_experiment():

    pos = 'r'
    # Seed sets.
    positive = ["easily","wonderfully","generously","happily","joyfully","marvellously","terifically","superbly","generously","best","better","joyously","jubilantly","merrily","correctly","advantageously","quickly"]
    negative = ["terribly","cruelly","angrily","sadly","wrongly"]
    neutral = ["administratively","financially","geographically","legislatively","managerial"]

    propagator = SentimentLexicon(positive, negative, neutral, pos, weight=0.2, rescale = False)
    
    sentiment_dict = propagator.iterate(runs=5)
    
    non_null = filter((lambda x : x[1] != 0.0), sentiment_dict.items())
    for key, val in sorted(non_null, key=itemgetter(1)):
        print "%s\t%s" % (key, val)
    print "Words with non-null scores:", len(non_null)

    pickle_out = open("sentiment_dict3.pickle", "wb")
    pickle.dump(sentiment_dict, pickle_out)
    pickle_out.close()

if __name__ ==  "__main__":
    tiny_adv_experiment()