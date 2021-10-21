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
    #First try to see if we can read the file in the correct format
    #First try to read a csv file
    try:
        df = pd.read_csv(table_file)
    except Exception as e:
        response = {"Message" : "Unable to Read CSV File {}".format(e)}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    #Now look for the column, which could be a name or numeric
    #Check for empty string
    if column == '':
        if len(df.columns) == 0:
            response = {"Message" : "Unable no columns in csv file"}
            status_code = status.HTTP_400_BAD_REQUEST
            return response, status_code
        if len(df.columns) > 1:
            response = {"Message" : "Specify a Single Column which contains the text"}
            status_code = status.HTTP_400_BAD_REQUEST
            return response, status_code
    #Now check to see if a column number was passed
    if str.isdigit(column):
        try:
            i_column = int(column)
            df = df[df.columns[i_column]]
        except Exception as e:
            response = {"Message" : "Unable to find column at position {}".format(str(i_column))}
            status_code = status.HTTP_400_BAD_REQUEST
            return response, status_code
    else:
        try:
            df = df[column]
        except Exception as e:
            response = {"Message" : "Unable to find column at position {}".format(column)}
            status_code = status.HTTP_400_BAD_REQUEST
            return response, status_code
    response = df
    status_code = status.HTTP_200_OK
    return response, status_code

#A function to perform porter stemming
def remove_stop_stem(text):
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    keepWords = []
    for r in words:
        if not r in stop_words:
            if r.isalpha():
                stemmed = porter.stem(r.lower())
                keepWords.append(stemmed)
    #convert it into a flat string
    kwstring = ' '.join(keepWords)
    return kwstring

#Now create a function to lemmatise and tokenize the text
def remove_stop_lem(text):
    stop_words = set(stopwords.words('english'))
    #words = text.split()
    words = word_tokenize(text)
    keepWords = []
    for r in words:
        if not r in stop_words:
            if r.isalpha():
                lemmatized = lemmatizer.lemmatize(r.lower())
                keepWords.append(lemmatized)
    #convert it into a flat string
    kwstring = ' '.join(keepWords)
    return kwstring 

#This example came from the SKlearn tutorial
#https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

def plot_top_words(model, feature_names, n_top_words, title,num_topics,topic_names):
    rows = math.ceil(num_topics/5)
    fs = 15*rows
    fig, axes = plt.subplots(rows, 5, figsize=(fs, fs), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        if len(topic_names) == 0:
            ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        else:
            topic_title = topic_names[topic_idx +1]
            ax.set_title(topic_title,
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    return plt    

#Train the NMF model
def train_nmf(df,num_topics):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(df)
    tfidf_feature_names=tfidf_vectorizer.get_feature_names()
    nmf = NMF(n_components=num_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
    n_top_words = 10
    plot_image = plot_top_words(nmf, tfidf_feature_names, n_top_words,
               'Topics in NMF model',num_topics,{})
    #Now return the vectorizer, the tfidf and the plot
    return tfidf_vectorizer, nmf, plot_image
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
    #Now create a model name based on these parameters
    model_name = filename + '_' + str(selected_column) + '_' + model_type + '_' + str(num_topics) + '_' + normalisation
    #Start by getting the text from the table
    response, status_code = get_table_text(full_file_path,selected_column)
    
    #Check the last function was ok
    if status_code != status.HTTP_200_OK:
        return response, status_code
    #If it was ok then it was a dataframe
    df = response.copy()
    #Now look to see if we need to do any stemming or Lemmatisation
    if normalisation == "Stemming":
        #Call the stemming function on the table
        df = df.apply(remove_stop_stem)
        response = df
    if normalisation == "Lemmatisation":
        #Call the lemmatisation function on the table
        df = df.apply(remove_stop_lem)
        response = df
    
    #Now look at the model start with NMF
    if model_type == 'NMF':
        #Check the num topics is > 0
        if num_topics <1:
            response = {"Message" : "Number of Topics must be greater than 0 for the NMF model"}
            status_code = status.HTTP_400_BAD_REQUEST
        #Now train the NMF
        tfidf_vectorizer, nmf, plot_image = train_nmf(df,num_topics)
        #Save the plot
        plot_image.savefig(model_name + '.png')
    return response, status_code

#Next picle model and save to external storage
    
