#Import the Required Libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from django.core.files.storage import FileSystemStorage
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

def train_table(request):
    response = {}
   
    #Get the File Object
    file_uploaded = request.FILES.get('file')
    #File Name
    name = file_uploaded.name
    #Save the file so it can be read
    fs = FileSystemStorage()
    filename = fs.save(name, file_uploaded)
    full_file_path = fs.location + "/" + filename
    
    #Check for the Model Type
    model_key_error = "model_type key error: LDA or NMF Model Selection Required"
    if 'model_type' not in request._full_data:
        response = {"Message":model_key_error}
        return response
    model_type = request._full_data['model_type']
    if model_type != 'LDA' and model_type != 'NMF':
        response = {"Message":model_key_error}
        return response
    
    #Check for the Selected Column
    selected_column_error = "selected column key missing"
    if 'selected_column' not in request._full_data:
        response = {"Message": selected_column_error}
        return response
    selected_column = request._full_data['selected_column']
    #Now check that the column can be extracted from the table

    #Check the Number of Topics
    #If Model Type is NMF, this cannot be blank
    if model_type == 'NMF' and 'num_topics' not in request._full_data:
        response = {"Message":"num_topics Key Error: Number of Topics is required for NMF Model"}
        return response
    #See if the number of topics is in the request
    if 'num_topics' in request._full_data:
        num_topics = request._full_data['num_topics']
        #See if it's an integer
        if str.isdigit(num_topics):
            num_topics = num_topics
            num_topics = int(num_topics)
        else:
            if num_topics == '':
                num_topics = 0
            else:
                response = {"Message" : "num_topics key error - expecting integer"}

    #Now get the Normalisation Method
    if 'normalisation' in request._full_data:    
        normalisation = request._full_data['normalisation']
        #Now check it's a valid value
        if normalisation != "None" and normalisation != "Stemming" and normalisation != "Lemmatisation":
            response = {"Message" : "normalisation Key Error - expecting None, Stemming or Lemmatisation"}
    else:
        normalisation = "None"
    return response

