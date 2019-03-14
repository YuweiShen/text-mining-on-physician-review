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
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from string import punctuation
from pymining import itemmining
from pattern.en import lemma


def _synonyms(seed,form):
    '''
    find the synonyms of the word with the help of wordnet
    input param seed: the word
    return: a list of synonyms of input
    '''
    synset = set()
    if form == 'n':
        try:
            wordset = wn.synset(seed+'.n.01')
        except:
            wordset = wn.synsets(seed)[0]
    elif form == 'v':
        try:
            wordset = wn.synset(seed+'.v.01')
        except:
            wordset = wn.synsets(seed)[0]
    else:
        try:
            wordset = wn.synset(seed+'.s.01')
        except:
            wordset = wn.synsets(seed)[0]
    for lemma in wordset.lemma_names():
        synset.add(lemma)
    for hypo in wordset.hyponyms():
        for lemma in hypo.lemma_names():
            synset.add(lemma)
    for hyper in wordset.hypernyms():
        for lemma in hyper.lemma_names():
            synset.add(lemma)
    synlist = list(synset)
    return(synlist)

def synonyms(seed,form=0):
    bigset = set()
    if form=='n':
        for word in _synonyms(seed,form):
            for elm in _synonyms(word,form):
                bigset.add(elm)
    else:
        for word in _synonyms(seed,form):
            bigset.add(word)
    return list(bigset)

#Read in data
cwd = 'C:\\Users\\58454\\Desktop\\Summer research'
reviewset = pd.read_csv(cwd+'\\reviews.csv')
reviews = list(reviewset['reviews'])

#preprocess
##convert the '_' in reviews back to ','
for i in range(len(reviews)):
    if reviews[i] is not np.nan:
        reviews[i] = reviews[i].replace('_',',')
## someone don't put a space before '.'    
for i in range(len(reviews)):
    if reviews[i] is not np.nan:
        reviews[i] = reviews[i].replace('.','. ')

tags=['VB','VBD','VBP','VBN','VBG','VBZ','NN','NNS'] #tags of noun and verb except proper nouns[专有名词]
wl = WL() #initialize the lemmatizer
stop = set(stopwords.words('english')) #a set of stop words
stop.update(punctuation) #add punctuations to the dataset
# remove some meaningless words
stop.update(["'s","'ve","'m",'wa','patient'])
stop.update(synonyms('doctor','n'))
stop.update(['go'])
stop.update(synonyms('recommend','v'))




# get frequent feature sets
# Data structure: why use set to store feature- eliminate redundency for each customer

def find_frequent(cur_doc_features):
    cur_doc_features = tuple(cur_doc_features)
    n_sentence = len(cur_doc_features)
    relim_input = itemmining.get_relim_input(cur_doc_features)# transform the input
    item_sets = itemmining.relim(relim_input, min_support=max(2,0.05*n_sentence))# Apriori rule mining; 0.05 is self-defined
    return item_sets# return feature of one doctor
   
all_features = []
doctors=[] #store the doctors
cur_doc = reviewset['doctor'][0]
cur_doc_features = []
num_doc_reviews = []
num=0 # number of the doctor's review (exclude nan)
for i in range(len(reviews)):# for each customer 
    if reviews[i] is np.nan:# if the reviews of a certain doctor is all nan, then append '[]'
        if (i<len(reviews)-1) and (reviewset['doctor'][i+1]!=reviewset['doctor'][i]):# mark the end of a doctor's reviews
            doctors.append(reviewset['doctor'][i])
            if cur_doc_features is []:
                all_features.append([])
                num_doc_reviews.append(0)
            else:
                all_features.append(find_frequent(cur_doc_features))
                num_doc_reviews.append(num)
                cur_doc_features = []
                num=0
                cur_doc = reviewset['doctor'][i+1]
        elif i==len(reviews)-1:
            doctors.append(reviewset['doctor'][i])
            all_features.append(find_frequent(cur_doc_features))
            num_doc_reviews.append(num)
    else:
        review = reviews[i]
        review_features = set() # if the current review and the first review belong to the same doctor and is not the last one, keep on mining features, otherwise start mining.
        sentences = nltk.sent_tokenize(review)
        for sentence in sentences: # for each sentence in review
            words = nltk.word_tokenize(sentence)
            word_tag = nltk.pos_tag(words) #tag the words
            for (word,tag) in word_tag: #select nouns and verbs in the sentences.
                if tag in tags:
                    lem = lemma(word)# lemmatize the words                   
                    if lem not in stop: # remove stop words
                        review_features.add(lem)
        review_features = tuple(review_features) # adjust for the association mining
        if reviewset['doctor'][i] == cur_doc: # do not change doctor
            num += 1
            cur_doc_features.append(review_features)
            if (i<len(reviews)-1) and (reviewset['doctor'][i] != reviewset['doctor'][i+1]):# at the end of one doctor's reviews
                doctors.append(reviewset['doctor'][i])
                all_features.append(find_frequent(cur_doc_features))  
                num_doc_reviews.append(num)             
                cur_doc_features = [] # delete the past doctor's information 
                num = 0
                cur_doc = reviewset['doctor'][i+1]              
        if i == len(reviews)-1 :
           doctors.append(reviewset['doctor'][i])
           all_features.append(find_frequent(cur_doc_features))      
           num_doc_reviews.append(num)
        
# now we got a list of features of 170 doctors stored in all_features   
# the all_features is a list of feature dictionary of all physicians   
# now calculate the frequency of each features

feature_frequency = dict() 
single_words = dict()
for i in range(170): # for each dotor
    item_dict = all_features[i]
    for featureset,freq in item_dict.items():
        if len(featureset) >= 2: # make sure we get phrases instead of single words
            if featureset not in feature_frequency :
                feature_frequency[featureset] = freq 
            else:
                feature_frequency[featureset] += freq 
        else:
            if featureset not in single_words:
                single_words[featureset] = freq 
            else:
                single_words[featureset] += freq 
            
# selct feature phrases whose frequency >=5 and sort the result
percentage = 0.01  
num_customer = sum(num_doc_reviews)
result = dict(filter(lambda x:x[1] >= percentage*num_customer,feature_frequency.items()))   
result = sorted(result.items(),key = lambda x:x[1],reverse = True)   
#select single word features whose frequency >=10 and sort the result   
single_result = dict()   
for singleset,freq in single_words.items():
   for word in singleset:
       flag = 1
       for (biset,freq2) in result:
            if word in biset:
                flag = 0
       if flag == 1:
            single_result[singleset] = freq
           
single_result = dict(filter(lambda x:x[1] >= percentage*num_customer,single_result.items()))   
single_result = sorted(single_result.items(),key = lambda x:x[1],reverse = True)  
    
# implicit words mining
# notice that 'doctor', 'dentist' are all frequent words, but contains little information
# select the adjective in the same sentence as 'dentist' etc.
all_adj = []
cur_doc = reviewset['doctor'][0]
cur_doc_adj = []
num=0 # number of the doctor's review (exclude nan)
for i in range(len(reviews)):# for each customer 
    if reviews[i] is np.nan:# if the reviews of a certain doctor is all nan, then append '[]'
        if (i<len(reviews)-1) and (reviewset['doctor'][i+1]!=reviewset['doctor'][i]):# mark the end of a doctor's reviews
            if cur_doc_adj is []:
                all_adj.append([])
            else:
                all_adj.append(find_frequent(cur_doc_adj))
                cur_doc_adj = []
                num=0
                cur_doc = reviewset['doctor'][i+1]
        elif i==len(reviews)-1:
            all_adj.append(find_frequent(cur_doc_adj))
    else:
        review = reviews[i]
        review_features = set() # if the current review and the first review belong to the same doctor and is not the last one, keep on mining features, otherwise start mining.
        sentences = nltk.sent_tokenize(review)
        for sentence in sentences: # for each sentence in review
            words = nltk.word_tokenize(sentence)
            word_tag = nltk.pos_tag(words) #tag the words
            for (word,tag) in word_tag: #select nouns and verbs in the sentences.
                if word in synonyms('doctor','n'): # make sure there are adjectives in the sentence
                    for (word1,tag1) in word_tag:
                        if tag1 == 'JJ':
                            review_features.add(word1)
                    break
        review_features = tuple(review_features) # adjust for the association mining
        if reviewset['doctor'][i] == cur_doc: # do not change doctor
            cur_doc_adj.append(review_features)
            if (i<len(reviews)-1) and (reviewset['doctor'][i] != reviewset['doctor'][i+1]):# at the end of one doctor's reviews
                all_features.append(find_frequent(cur_doc_adj))               
                cur_doc_adj = [] # delete the past doctor's information 
                cur_doc = reviewset['doctor'][i+1]              
        if i == len(reviews)-1 :
           all_adj.append(find_frequent(cur_doc_adj))      

adj_frequency = dict()
for i in range(len(all_adj)): # for each dotor, calculate the adj's to describe a doctor
    adj_dict = all_adj[i]
    for featureset,freq in adj_dict.items():
            if featureset not in adj_frequency :
                adj_frequency[featureset] = freq 
            else:
                adj_frequency[featureset] += freq 
# then we can have a look at the adjectives used to desribe the doctors.                
adj_frequency
# merge and calculate the final features
# form the initial feature list subjectively 
feature_map=dict()
feature_map['bedside manner']=[('manner'),
           ('feel','make'),('comfortable'),('kind'),
           ('funny'),('friendly'),('gentle'),
           ('answer'),('explain')]
feature_map['diagnosis'] = [('diagnosis'),('treatment'),('thorough')]
feature_map['service time'] = [('take','time'),('service'),('procedure')]
feature_map['waiting time'] = [('wait'),('slow')]
feature_map['insurance'] = [('insurance')]
feature_map['knowledge'] = [('knowledgeable'),('professional'),('informative')]
feature_map['office'] = [('office'),('clean'),('place')]

# now use WordNet to enlarge the list
for (key,items) in feature_map.items():
    potential = set()
    for tup in items:
        if len(tup)!=2: # not a phrase
            tag = nltk.pos_tag(nltk.word_tokenize(tup))[0][1][0]
            tag = tag.lower()
            potential.update( _synonyms(tup,tag) )
    potential = list(potential)
    feature_map[key] += potential

# calculate the frequency of features for each doctor

doc_features = [] # every element is a list corresponding to one doctor

def cumu_cal_frequency(featuredic,review,cumudict):
    '''
    calculate whether a feature appears for one customer
    input params: featuredic - a feature and corresponding words
                  review - review of a specific customer
    return a feature list made up of 1 or 0
    '''
    if review is np.nan:
        return 
    else:
        for (feature,items) in feature_map.items():# for each feature
            count = 0 
            sentences = nltk.sent_tokenize(review)
            new = []
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
                for word in words:
                    new.append(lemma(word))
            for item in items:
                if len(item) == 2: # item is a phrase
                    if item[0] in new and item[1] in new:
                        count = 1
                else:
                    if item in new:
                        count = 1                          
            cumudict[feature] += count # add the number to corresponding feature
        return cumudict

cur_doctor = reviewset['doctor'][0]  
cumudict = dict()
for f in feature_map.keys(): # initialize the dictionary
    cumudict[f] = 0     
doc_frequency = []             
for i in range(len(reviews)):# for each customer 
    review = reviews[i]
    if reviewset['doctor'][i] == cur_doctor: # do not change doctor
        cumu_cal_frequency(feature_map,review,cumudict) # calculate the cumulative frequency
    else:# now a new doctor
        doc_frequency.append(cumudict) # add the last cumudict
        cur_doctor = reviewset['doctor'][i]
        cumudict = dict() 
        for f in feature_map.keys(): # initialize the dictionary
            cumudict[f] = 0
        cumu_cal_frequency(feature_map,review,cumudict)   
    if i == len(reviews)-1:
        doc_frequency.append(cumudict) 

# change the list to a table
data = pd.DataFrame(data = doc_frequency, index = doctors)
# merge the number of customer reviews
data['number of reviews'] = num_doc_reviews
data.to_excel('result.xlsx')