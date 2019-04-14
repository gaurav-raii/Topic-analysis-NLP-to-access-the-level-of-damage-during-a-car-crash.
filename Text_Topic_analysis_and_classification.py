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
