#Import the Required Libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
#NLTK for speach
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
#nltk.download('wordnet')
#download pos
#nltk.download('averaged_perceptron_tagger')
#nltk.download('tagsets')
import re
#Scikit for learning mechanisms
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

def list_options(request):
    options = {
        "file" : "Input File",
        "selected_column" : "The Column that Contains the Text",
        "model_type" : "LDA or NMF",
        "num_topics" : "Number of Topics - madatory for NMF",
        "normalisation" : "Stemming, Lemmatisation or None"
    }
    return options

