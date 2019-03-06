# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:19:48 2019

@author: Yuwei Shen
"""
import pandas as pd
import numpy as np
import os
import nltk 
from nltk.stem import WordNetLemmatizer as WL
from nltk.corpus import stopwords
from string import punctuation
from pymining import itemmining,assocrules

#Read in data
cwd=os.getcwd()
reviewset=pd.read_csv(cwd+'\\reviews.csv')
reviews=list(reviewset['reviews'])

#preprocess
##convert the '_' in reviews back to ','
for i in range(len(reviews)):
    if reviews[i] is not np.nan:
        reviews[i]=reviews[i].replace('_',',')

tags=['VB','VBD','VBP','VBN','VBG','VBZ','NN','NNS'] #tags of noun and verb except proper nouns[专有名词]
wl = WL() #initialize the lemmatizer
stop=set(stopwords.words('english')) #a set of stop words
stop.update(punctuation) #add punctuations to the dataset
stop.update(["'s","'ve"])

# get frequent feature sets
# Data structure: why use set to store feature- eliminate redundency for each customer

def find_frequent(cur_doc_features):
    cur_doc_features=tuple(cur_doc_features)
    n_sentence=len(cur_doc_features)
    relim_input = itemmining.get_relim_input(cur_doc_features)# transform the input
    item_sets = itemmining.relim(relim_input, min_support=max(2,0.05*n_sentence))# Apriori rule mining; 0.05 is self-defined
    return item_sets# return feature of one doctor

     
all_features=[]
past_doc=reviewset['doctor'][0]
cur_doc_features=[]
j=1
for i in range(len(reviews)):# for each customer   
    if reviews[i] is not np.nan: 
        review=reviews[i]
        review_features=set() # if the current review and the first review belong to the same doctor and is not the last one, keep on mining features, otherwise start mining.
        sentences=nltk.sent_tokenize(review)
        for sentence in sentences: # for each sentence in review
            words=nltk.word_tokenize(sentence)
            word_tag=nltk.pos_tag(words) #tag the words
            for (word,tag) in word_tag: #select nouns and verbs in the sentences.
                if tag in tags:
                    pos=tag[0].lower()
                    lem=wl.lemmatize(word,pos=pos)# lemmatize the words
                    lem=lem.lower()# convert to lowercase   
                    if lem not in stop: # remove stop words
                        review_features.add(lem)
        review_features=tuple(review_features) # adjust for the association mining
        if reviewset['doctor'][i]==past_doc: # the same as the past doctor
            cur_doc_features.append(review_features)
        if (reviewset['doctor'][i]!=past_doc) or i==len(reviews)-1: # at the end of one doctor's reviews
            j+=1
            print(j)
            all_features.append(find_frequent(cur_doc_features)          
            cur_doc_features=[] # delete the past doctor's information
            cur_doc_features.append(review_features)
            past_doc=reviewset['doctor'][i] # refresh the doctor mark
            # now we got a list of features of 170 doctors stored in all_features     
        

# the all_features is a list of feature dictionary of all physicians   
# now calculate the frequency of each features
feature_frequency=dict() 
single_words=dict()
for item_dict in all_features:   
    for featureset,freq in item_dict.items():
        if len(featureset)>=2: # make sure we get phrases instead of single words
            if featureset not in feature_frequency:
                feature_frequency[featureset]=freq
            else:
                feature_frequency[featureset]+=freq
        else:
            if featureset not in single_words:
                single_words[featureset]=freq
            else:
                single_words[featureset]+=freq

             
# selct feature phrases whose frequency >=5 and sort the result             
result=dict(filter(lambda x:x[1]>=5,feature_frequency.items()))   
result = sorted(result.items(),key = lambda x:x[1],reverse = True)   
#select single word features whose frequency >=10 and sort the result   
single_result=dict()   
for singleset,freq in single_words.items():
   for word in singleset:
       flag=1
       for (biset,freq2) in result:
            if word in biset:
                flag=0
       if flag==1:
            single_result[singleset]=freq
single_result=dict(filter(lambda x:x[1]>=10,single_result.items()))   
single_result = sorted(single_result.items(),key = lambda x:x[1],reverse = True)              
'''           
## if not divide it into different doctors
all_sent_features=[]
stem=nltk.PorterStemmer()
for review in reviews:
    if review is not np.nan:
        sentences=nltk.sent_tokenize(review)
        for sentence in sentences:
            sent_features=[]
            words=nltk.word_tokenize(sentence)
            words_tag=nltk.pos_tag(words)
            for (word,tag) in words_tag:
                if tag in tags:
                    pos=tag[0].lower()
                    lem=wl.lemmatize(word,pos)
                    if lem not in stop:
                        sent_features.append(lem)
            if sent_features is not []:
                all_sent_features.append(tuple(sent_features))

nsentence=len(all_sent_features)                    
trans=tuple(all_sent_features)
rinput = itemmining.get_relim_input(trans)# transform the input
items = itemmining.relim(rinput, min_support=0.01*nsentence)# Apriori rule mining; 0.05 is self-defined
'''
# contruct mapping 
# Use PMI to classify implicit features
# the explicit set can be treated as seed feature set.  we can generate synonyms with the help of corpus
# then calculate PMI for each word in the sentence. Choose the highest one.
# bedside_manner=[frosenset{'take','time'},]
# service_time =[]
# 
