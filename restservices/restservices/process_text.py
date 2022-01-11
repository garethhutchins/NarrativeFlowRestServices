from rest_framework import response, status
import requests
from django.conf import settings as conf_settings
import json
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
#nltk.download('wordnet')
#download pos
#nltk.download('averaged_perceptron_tagger')
#nltk.download('tagsets')
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from .common_processing import remove_stop_lem, remove_stop_stem
import pickle
import urllib.request
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


#A basic function to list the inputs
def list_train_table_options(request):
    options = {
        "text" : "The raw text to be processed",
        "window_size" : "The Size of the sliding window to analyse the text",
        "window_slide" : "The size of the increase between itterations of the text classification",
        "model_id" : "The model id / name used to classify the text in the window size"
    }
    status_code = status.HTTP_200_OK
    return options, status_code

#Now predict the text that was submitted
def predict_text(request):
     #Get the model from the provided model id
    #get the model id from the request
    if 'model_id' not in request.data:
        response = {'Message':'No Model ID Provided'}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    # See if the setting exists if not create it and leave it blank
    if hasattr(conf_settings, 'PERSISTENT_STORAGE'):
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    else:
        response = {"Message" : "Persistent Storage URI has not been configured please configure"}
        status_code = status.HTTP_424_FAILED_DEPENDENCY
        return response, status_code
   
   #Check to see that there was text in the request
    if 'text' not in request.data:
        response = {'Message':'No Text Provided'}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    #Check to see if the Window Size was in the request
    if 'window_size' not in request.data:
        response = {'Message':'No window_size Provided'}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    window_size = int(request.data['window_size'])
    if 'window_slide' not in request.data:
        response = {'Message':'No window_slide Provided'}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    window_slide = int(request.data['window_slide'])
    #Now get the model that was in the request
    model_id = request.data['model_id']
    persistent_storage_models = persistent_storage + "/storage/" + model_id
    models = requests.get(persistent_storage_models)
    models_json = json.loads(models.content)
    #Get the files from the storage service
    saved_model = pickle.load(urllib.request.urlopen(models_json['save_model']))
    
    #Now split the text into the provided window size and slide
    text = request.data['text']
    words = word_tokenize(text)
    num_words = len(words)
    pos = 0
    topics = []
    scores = []
    text_window_contents = []
    while pos < (num_words - window_size):
        current_block = ' '.join(words[pos:(pos+window_size)])
        #First check to see if it's the K-Means model
        if saved_model['model_type'] == 'K-MEANS':
            #This has a different process as we need to do word embedding
            embedding_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            embeddings = embedding_model.encode(current_block)
            k_class = saved_model['model'].classify(embeddings)
        if saved_model['normalisation'] == 'Lemmatisation':
            current_block = remove_stop_lem(current_block)
        if saved_model['normalisation'] == 'Stemming':
            current_block = remove_stop_stem(current_block)
        tf_test = saved_model['vectorizer'].transform([current_block])
        #See if it's a TF-IDF model as the command is different
        if saved_model['model_type'] == 'TF-IDF':
            #Get the probabbilities of the classes
            predictions = saved_model['model'].predict_proba(tf_test)
            #Get the class names
            dict_vals = saved_model['model'].classes_
        else:
            predictions = saved_model['model'].transform(tf_test)
            dict_labels = models_json['topic_labels']
            dict_vals = dict_labels.values()
        tp = predictions*100
        tp = np.around(tp,decimals=2)
        predictionsx = pd.DataFrame(tp).T
        
        labels = pd.DataFrame(dict_vals)
        window_pred = pd.concat([labels, predictionsx],axis=1)
        window_pred.columns = ['Topic','Confidence']
        window_pred = window_pred.sort_values(by=['Confidence'],ascending=False)
        #Add the confidence score to the output
        topics.append(window_pred.iloc[:,0])
        scores.append(window_pred.iloc[:,1])
        text_window_contents.append(' '.join(words[pos:(pos+window_size)]))
        pos += window_slide
    f_results = pd.DataFrame({'Text':text_window_contents,'Topics':topics,'Scores':scores})
    json_results = f_results.to_json(orient='records')
    status_code = status.HTTP_200_OK
    return json.loads(json_results), status_code

