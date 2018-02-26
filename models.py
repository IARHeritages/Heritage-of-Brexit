'''
Created on Sep 20, 2017

Module created to apply topic modelling.


@author: 
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
    Method to get the text output from results in a CSV. Retrieves relevant texts only.
    @param pn the path to find the relevant text
    '''
def retrieveText(self,pn):
    del self.listResults[:]
        
    doc_set=[]
    os.chdir(pn+'/output')
    en_stop = stops
    result=[]
    for filename in os.listdir(os.getcwd()):
        txt=''
        if(filename == ".DS_Store" or "lda" in filename or "hdp" in filename or ".csv" not in filename):
            continue
        print(filename)
        with open(filename, 'rU') as csvfile:
            reader = csv.reader(csvfile, quotechar='|') 
                
            i=0
            try:
                for row in reader:
                    if row in ['\n', '\r\n']:
                        continue;
                    if(i==0):
                        i=i+1
                        continue
                    if(len(row)<1):
                        continue
                        
  #                      if i==100:
  #                          break
                    text=''
                    for r in row:
                        text+=r.strip()
                            
                    text=re.sub('"','',text)
                    text=re.sub(',','',text)
                    text.strip()
                    tFalse=True
                        
                    if(len(result)==0):
                        result.append(text)
                        i+=1
                        txt=txt+" "+text
                            
                    if(tFalse==True):
                            txt=txt+" "+text
                             
                            if text==' ':
                                continue
                             
                            tokenizer = RegexpTokenizer(r'\w+')
                            text = tokenizer.tokenize(unicode(text, errors='replace'))
                            stopped_tokens = [t for t in text if not t in en_stop]
                             
                            doc_set.append(stopped_tokens)  
                    i+=1 
            except csv.Error, e:
                sys.exit('line %d: %s' % (reader.line_num, e))
            
            
            
        return doc_set
'''
Data files loaded from test

@return a file for training or testing.
'''

def test_directories():
    test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
    lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
    
    with open(lee_train_file) as f:
        for n, l in enumerate(f):
            if n < 5:
                print([l])

    return lee_train_file

''''
Create the text for analysis from the filename

@param fname the file name
'''
def build_texts(fname):
    """
    Function to build tokenized texts from file
    
    Parameters:
    ----------
    fname: File to be read
    
    Returns:
    -------
    yields preprocessed line
    """
    with open(fname) as f:
        for line in f:
            yield gensim.utils.simple_preprocess(line, deacc=True, min_len=3)

'''
Process text files based on minimum length of file

@param files for processing
'''
def preProcsText(files):
  
        for f in files:
            yield gensim.utils.simple_preprocess(f, deacc=True, min_len=3)

'''
Method for text processing with input text.

@param input texts
@return texts the texts to be returned after processing.
'''
def process_texts(texts):
    """
    Function to process texts. Following are the steps we take:
    
    1. Stopword Removal.
    2. Collocation detection.
    3. Lemmatization (not stem since stemming can reduce the interpretability).
    
    Parameters:
    ----------
    texts: Tokenized texts.
    
    Returns:
    -------
    texts: Pre-processed tokenized texts.
    """
    
    # reg. expression tokenizer
        
    texts = [[word for word in line if word not in stops] for line in texts]
    texts = [bigram[line] for line in texts]
    texts = [[word.split('/')[0] for word in lemmatize(' '.join(line), allowed_tags=re.compile('(NN)'), min_length=3)] for line in texts]

    return texts

'''
Method for using a coherence model to look at topic coherence for LDA models.

@param dictionary the dictionary of assessment 
@param corpus the texts
@param limit the limit of topics to assess
@return lm_list lda model output
@return c_v coherence score
''' 
def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        del cm
            
    return lm_list, c_v

"""
Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
evaluating the topic model until this threshold is crossed. 

@return lm: Final evaluated topic model
@return top_topics: ranked topics in decreasing order. List of tuples
"""
def ret_top_model():
    
    top_topics = [(0, 0)]
    while top_topics[0][1] < 0.97:
        lm = LdaModel(corpus=corpus, id2word=dictionary)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=train_texts, dictionary=dictionary, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
    return lm, top_topics

'''Method to add results and clean.
@param result and text to clean
   
@return result_dict dictionary of the term and values'''
def addTotalTermResults(t):
    result_dict={}
    for a,b in t:
            text=re.sub('"',"",b)
            text.replace(" ","")
            txts=text.split("+")
            for t in txts:
                ttnts=t.split("*")
                v=float(ttnts[0])
                t=ttnts[1]
                t=str(a)+":"+t
                if(t in result_dict):
                    continue
                else:
                    t=t.strip()
                    result_dict[t]=v 
                           
    return result_dict
                        
'''Add dictionary to a list of results from each text
    @param result_dict this is the resulting terms'''        
def addToResults(result_dict):
        listResults.append(result_dict)
            
        
'''Method aggregates all the dictionaries for keyterms and their values.
    @return dct a dictionary of all keyterms and values'''           
def dictionaryResults():
    #set the dictionary
    dct={}
        
    #iterate over all tweets and add to total dictionary
    for dictionary in listResults:
            for key in dictionary:
                    
                v=dictionary[key]
                if(key in dct):
                    vv=dct[key]
                    vv=v+vv
                    dct[key]=vv
                else:
                    dct[key]=v 
                        
    return dct
    
'''Output results of the analysis
@param nn the number of topics used for the output name
@param i topic number
@param model the model
'''
def printResults(i,model):
        
  #     os.chdir('../')
        pn=os.path.abspath(__file__)
        pn=pn.split("/iScraper/")[0]+'/iScraper/'+model
        
        filename=pn+'/'+model+'_results'+"-"+str(i)+'-'+'.csv'
        
        fieldnames = ['Topic','Term','Value']
        
        dct=dictionaryResults()
        with open(filename, 'wb') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)

            writer.writeheader()
            
            for key in dct:
                v=dct[key]
                tn=key.split(":")[0]
                kt=key.split(":")[1]
                writer.writerow({'Topic':str(tn),'Term': str(kt.encode("utf-8")),'Value':str(v)})
        
'''Method to print csv output results of the evaluations conducted 

@param modList the model evaluated
@param results the result scores
@i the index output desired
'''
def printEvaluation(modList,results,i):
    pn=os.path.abspath(__file__)
    pn=pn.split("/iScraper/")[0]+'/iScraper/'
        
    filename=pn+'/evaluation'+"-"+str(i)+'-'+'.csv'
        
    fieldnames = ['Model','Score']
    
    with open(filename, 'wb') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(0,len(modList)):
        
                writer.writerow({'Model':str(modList[i]),'Score': str(results[i])})
        
#lee_train_file=test_directories()
#train_texts = list(build_texts(lee_train_file))

#bigram = gensim.models.Phrases(train_texts)


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
    #lsi model
    
    lsimodel = LsiModel(corpus=corpus, num_topics=i, id2word=dictionary)

    lsitopics=lsimodel.show_topics(num_topics=i)

    result_dict=addTotalTermResults(lsitopics)    
    addToResults(result_dict)
    printResults(i,'lsi')
    
    del listResults[:]    
    
    #hdp model
    hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)

    hdpmodel.show_topics()

    hdptopics = hdpmodel.show_topics(num_topics=i)

    result_dict=addTotalTermResults(hdptopics)
            
    #add results to total kept in a list     
    addToResults(result_dict)
    
    printResults(i,'hdp')
    del listResults[:] 
     
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
