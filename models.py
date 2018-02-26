###Module created to apply topic modelling.
###Project: Ancient Identities in Modern Britain (IARH) project; ancientidentities.org
###Author: Mark Altaweel, based on initial code shared at:
###https://markroxor.github.io/gensim/static/notebooks/gensim_news_classification.html#topic=1&lambda=1&term=

'''


import os
import re
from hdp import HDP
import operator
#import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np

import sys
import csv
from nltk.tokenize import RegexpTokenizer
import pyLDAvis.gensim

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint
from gensim.utils import lemmatize
from nltk.corpus import stopwords
import pyLDAvis.gensim

warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
stops = set(stopwords.words('english'))  # nltk stopwords list
listResults=[]  # list of results to hold

'''
The below code is executed to conduct the models for topic modelling and 
coherence testing for LDA models
'''
pn=os.path.abspath(__file__)
pn=pn.split("src")[0]
results=retrieveText(pn)

bigram = gensim.models.Phrases(results) 
#train_texts = process_texts(train_texts)

train_texts=process_texts(results)


preProcsText(results)

dictionary = Dictionary(train_texts)
corpus = [dictionary.doc2bow(text) for text in train_texts]

#up to 50 topics are tested 
for i in range(1,50,1):
    
    #lda model
    ldamodel = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary)
    num=str(i)
    ldamodel.save('lda'+num+'.model')
    ldatopics = ldamodel.show_topics(num_topics=i)
    
    result_dict=addTotalTermResults(ldatopics)    
    addToResults(result_dict)
    printResults(i,'lda')
    
    del listResults[:] 
    visualisation = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization.html') 

#coherence model evaluation
lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=train_texts, limit=i)

#lm, top_topics = ret_top_model()

#coherence model results
printEvaluation(lmlist,c_v,i)
