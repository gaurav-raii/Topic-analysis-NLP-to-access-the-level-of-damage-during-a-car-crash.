# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:46:19 2019

@author: gaura
"""

from AdvancedAnalytics import TextAnalytics
import pandas as pd
import numpy as np

# Classes for Text Preprocessing 
import string
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet as wn 
from nltk.corpus import stopwords

# sklearn methods for Preparing the Term-Doc Matrix 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer

# sklearn methods for Extracting Topics using the Term-Doc Matrix 
from sklearn.decomposition import LatentDirichletAllocation 
from sklearn.decomposition import TruncatedSVD 
from sklearn.decomposition import NMF

nltk.download('punkt') 
nltk.download('averaged_perceptron_tagger') 
nltk.download('stopwords') 
nltk.download('wordnet')

# Increase column width to let pandy read large text columns 
pd.set_option('max_colwidth', 32000)

# Read California Chardonnay Reviews 
df = pd.read_excel("C:/Users/gaura/Desktop/stat 656/week 11/week 11 assignment/GMC_Complaints.xlsx")

# Setup program constants and reviews 
n_reviews = len(df['description']) 
s_words = 'english' 
ngram = (1,2) 
reviews = df['description']

# Constants 
m_features = None # default is None 
n_topics = 8 # number of topics 
max_iter = 10 # maximum number of iterations 
max_df = 0.5 # max proportion of docs/reviews allowed for a term 
learning_offset = 10. # default is 10 
learning_method = 'online' # alternative is 'batch' for large files 
tf_matrix='tfidf'

# Create the Review by Term Frequency Matrix using Custom Analyzer 

ta = TextAnalytics() 
cv = CountVectorizer(max_df=max_df, min_df=2, max_features=m_features,analyzer=ta.my_analyzer) 
tf = cv.fit_transform(reviews) 
terms = cv.get_feature_names() 
print('{:.<22s}{:>6d}'.format("Number of Reviews", len(reviews))) 
print('{:.<22s}{:>6d}'.format("Number of Terms", len(terms)))

term_sums = tf.sum(axis=0) 
term_counts = [] 
for i in range(len(terms)):
    term_counts.append([terms[i], term_sums[0,i]]) 
def sortSecond(e):
    return e[1] 
term_counts.sort(key=sortSecond, reverse=True) 
print("\nTerms with Highest Frequency:") 
for i in range(10):
    print('{:<15s}{:>5d}'.format(term_counts[i][0], term_counts[i][1])) 

 # Construct the TF/IDF matrix from the Term Frequency matrix 
 print("\nConstructing Term/Frequency Matrix using TF-IDF") 
 # Default for norm is 'l2', use norm=None to supress 
 tfidf_vect = TfidfTransformer(norm=None, use_idf=True) #set norm=None 
 # tf matrix is (n_reviews)x(m_terms) 
 tf = tfidf_vect.fit_transform(tf)
# Display the terms with the largest TFIDF value 
term_idf_sums = tf.sum(axis=0) 
term_idf_scores = [] 
for i in range(len(terms)):
    term_idf_scores.append([terms[i], term_idf_sums[0,i]]) 
print("The Term/Frequency matrix has", tf.shape[0], " rows, and", tf.shape[1], " columns.") 
print("The Term list has", len(terms), " terms.") 
term_idf_scores.sort(key=sortSecond, reverse=True) 
print("\nTerms with Highest TF-IDF Scores:") 
for i in range(10):
    j = i
    print('{:<15s}{:>8.2f}'.format(term_idf_scores[j][0],  term_idf_scores[j][1]))
    
uv = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter,\
                               learning_method=learning_method, \
                               learning_offset=learning_offset, \
                                random_state=12345) 
U = uv.fit_transform(tf) 

# Display the topic selections 
print("\n********** GENERATED TOPICS **********") 
TextAnalytics.display_topics(uv.components_, terms, n_terms=15, mask=None)
# Store topic selection for each doc in topics[] 
topics = [0] * n_reviews 
for i in range(n_reviews):
    max = abs(U[i][0])
    topics[i] = 0
    for j in range(n_topics):
        x = abs(U[i][j])
        if x > max:
            max = x
            topics[i] = j
            
#*** Store Topic Scores in rev_score *** 
rev_scores = [] 
for i in range(n_reviews):
     u = [0] * (n_topics+1)
     u[0] = topics[i]
     for j in range(n_topics):
         u[j+1] = U[i][j]
     rev_scores.append(u)

# Setup review topics in new pandas dataframe 'df_rev' 
cols = ["topic"] 
for i in range(n_topics):
    s = "T"+str(i+1)
    cols.append(s) 
df_rev = pd.DataFrame.from_records(rev_scores, columns=cols) 
df = df.join(df_rev)

# checking for null values
df[['nthsa_id','Year','make','model','abs','mileage','topic']].isnull().sum(axis='index')
 
#df=df.dropna() ( if we dont want to impute we can just uncomment this )

# imputing mileage and abs columns as only they have missing values
df['mileage'] = df['mileage'].fillna(df['mileage'].mean())
mode = df['abs'].mode()
df['abs'] = df['abs'].fillna('N')

#Dropping duplicate and unwanted Columns 
df= df.drop(columns=['nthsa_id','description','topic'])

#normalizing mileage column
df['mileage'] = df['mileage']  / (df['mileage'].max() - df['mileage'].min())

# encoding the categorical data
def my_encoder(z):
    for i in z:
        a=df[i][df[i].notnull()].unique()
        for col_name in a:
            df[i+'_'+str(col_name)]= df[i].apply(lambda x: 1 if x==col_name else 0)

categorical = ['Year','make','model','abs','crashed']
my_encoder(categorical)
df= df.drop(columns=categorical)

# making arrays of predictor variables and target variable
X= df.drop(columns=['crashed_Y','crashed_N'])
Y= df['crashed_Y'].tolist()