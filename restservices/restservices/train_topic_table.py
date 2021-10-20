#Import the Required Libraries
import sys
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

from rest_framework import status

#Get the text from the table file
def get_table_text(table_file,column):
        try:
            df = pd.read_csv(table_file)
            response = df
        except Exception as e:
            response = {"Message" : "Unable to Read CSV File {}".format(e)}
            status_code = status.HTTP_400_BAD_REQUEST
            return response, status_code
        status_code = status.HTTP_200_OK
        return response, status_code


#A basic function to list the inputs
def list_options(request):
    options = {
        "file" : "Input File",
        "selected_column" : "The Column that Contains the Text",
        "model_type" : "LDA or NMF",
        "num_topics" : "Number of Topics - madatory for NMF",
        "normalisation" : "Stemming, Lemmatisation or None"
    }
    return options

#The Train Table function to train topics based on table contents
def train_table(request):
   #Check that there is a file:
    if len(request.FILES) == 0:
        response = {"Message" : "Missing File from Request"}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    #Get the File Object
    file_uploaded = request.FILES.get('file')
    #File Name
    name = file_uploaded.name
    #Save the file so it can be read
    fs = FileSystemStorage()
    filename = fs.save(name, file_uploaded)
    full_file_path = fs.location + "/" + filename

    #Check for the Selected Column
    selected_column_error = "selected column key missing"
    if 'selected_column' not in request._full_data:
        response = {"Message": selected_column_error}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    selected_column = request._full_data['selected_column']
    #Now check that the column can be extracted from the table
    
    #Check for the Model Type
    model_key_error = "model_type key error: LDA or NMF Model Selection Required"
    if 'model_type' not in request._full_data:
        response = {"Message":model_key_error}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    model_type = request._full_data['model_type']
    if model_type != 'LDA' and model_type != 'NMF':
        response = {"Message":model_key_error}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    
    

    #Check the Number of Topics
    #If Model Type is NMF, this cannot be blank
    if model_type == 'NMF' and 'num_topics' not in request._full_data:
        response = {"Message":"num_topics Key Error: Number of Topics is required for NMF Model"}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
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
                status_code = status.HTTP_400_BAD_REQUEST
                return response, status_code

    #Now get the Normalisation Method
    if 'normalisation' in request._full_data:    
        normalisation = request._full_data['normalisation']
        #Now check it's a valid value
        if normalisation != "None" and normalisation != "Stemming" and normalisation != "Lemmatisation":
            response = {"Message" : "normalisation Key Error - expecting None, Stemming or Lemmatisation"}
            status_code = status.HTTP_400_BAD_REQUEST
            return response, status_code
    else:
        normalisation = "None"
    #If we get this far then things are OK
    #Start by getting the text from the table
    response, status_code = get_table_text(full_file_path,selected_column)
    return response, status_code

    
