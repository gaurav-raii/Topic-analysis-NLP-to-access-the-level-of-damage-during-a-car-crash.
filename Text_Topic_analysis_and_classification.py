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
# max_df is a limit for terms. If a term has more than this 
# proportion of documents then that term is dropped. Use max_df=1.0 
# to eliminate this behavior. Typical values are max_df between 0.5 and 0.95

