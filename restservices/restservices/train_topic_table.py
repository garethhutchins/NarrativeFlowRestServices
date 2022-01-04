#Import the Required Libraries
import sys
import pandas as pd
import numpy as np
import math
import os

from django.core.files.storage import FileSystemStorage

import requests 

#Save and load the settings
from django.conf import settings as conf_settings

from rest_framework import status

from django.conf import settings as conf_settings

#Use pickle for saving
import pickle
import uuid
import json

#Use the commong functions
from .common_processing import remove_stop_lem, train_nmf, remove_stop_stem, train_lda, predict_lda_topics, train_tfidf, train_kmeans


#Get the text from the table file
def get_table_text(table_file,column):
    #Before we do anything with the data, we need to make sure that all of the settings have been configured
    persistent_storage = ""
    if hasattr(conf_settings, 'PERSISTENT_STORAGE'):
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    #We now need to see if it's blank, if so we need to return an error    
    if persistent_storage == "":
        response = {"Message":"Configure Model Saving Settings in ./service_settings"}
        status_code = status.HTTP_428_PRECONDITION_REQUIRED
        return response, status_code
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

#A basic function to list the inputs
def list_train_table_options(request):
    options = {
        "file" : "Input File",
        "selected_column" : "The Column that Contains the Text",
        "model_type" : "k-means, TF-IDF, LDA or NMF",
        "num_topics" : "Number of Topics - madatory for NMF & k-means",
        "normalisation" : "Stemming, Lemmatisation or None",
        "label_column" : "The Labels for TF-IDF Training" 
    }
    return options

#The Train Table function to train topics based on table contents
def train_table(request):
   #Check that there is a file:
    file_uploaded = request.FILES.get('file')
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
    model_key_error = "model_type key error: k-means, LDA, NMF or TF-IDF Model Selection Required"
    if 'model_type' not in request._full_data:
        response = {"Message":model_key_error}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    model_type = request._full_data['model_type']
    if model_type != 'LDA' and model_type != 'NMF' and model_type != 'TF-IDF' and model_type != 'k-means':
        response = {"Message":model_key_error}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    
    #Check the Number of Topics
    #If Model Type is NMF, this cannot be blank
    if model_type == 'NMF' and 'num_topics' not in request._full_data:
        response = {"Message":"num_topics Key Error: Number of Topics is required for NMF Model"}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    #If Model Type is k-means, this cannot be blank
    if model_type == 'k-means' and 'num_topics' not in request._full_data:
        response = {"Message":"num_topics Key Error: Number of Topics is required for NMF Model"}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    #See if the number of topics is in the request
    num_topics = 0
    if 'num_topics' in request._full_data:
        num_topics = request._full_data['num_topics']
        #See if it's an integer
        if str.isdigit(num_topics):
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
    #If the selected Model is k-means then we're going to do the normalisation after the creation of the model
    

    if normalisation == "Stemming" and model_type != 'k-means':
        #Call the stemming function on the table
        df = df.apply(remove_stop_stem)
        response = df
    if normalisation == "Lemmatisation" and model_type != 'k-means':
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
        model = nmf
        vectorizer = tfidf_vectorizer
        #Save the plot
        plot_image.savefig(model_name + '.png')
    #Now see if the model is LDA
    if model_type == 'LDA':
        #See if the number of topics has been provided
        if num_topics <1:
            #If this is the case then we'll try and predict the number of topics
            num_topics = predict_lda_topics(df)
        
        #Train the LDA
        tf_vectorizer, lda, plot_image = train_lda(df,num_topics)
        model = lda
        vectorizer = tf_vectorizer
        #Save the plot
        plot_image.savefig(model_name + '.png')
    #Now check to see if the Model Type is k-means
    if model_type == 'k-means':
        model, plot_image = train_kmeans(df,num_topics)
    #Now see if the model type is TF-IDF
    #initialize this to empty as we'll use it later
    tfidf_labels = {}
    if model_type == "TF-IDF":
        #Now check to see if the label column has been provided
        if 'label_column' in request._full_data:  
            label_column = request._full_data['label_column']
            labels, status_code = get_table_text(full_file_path,label_column)
            if status_code != 200:
                return labels, status_code
            else:
                #Now we need to train the TF-IDF model
                vectorizer, model, score, plot_image, lbls = train_tfidf(df,labels)
                plot_image.savefig(model_name + '.png')
                tfidf_labels = {'score': score,'labels':json.dumps(lbls)}
                num_topics = len(labels.unique())
        else:
            response = {"Message" : "Label Column requires for TF-IDF training"}
            status_code = status.HTTP_400_BAD_REQUEST
            return response, status_code
    


    #Now save the model
    saved_model = {'file_name':filename,
                    'selected_column':selected_column,
                    'model_type':model_type,
                    'normalisation':normalisation,
                    'vectorizer':vectorizer,
                    'model':model}
    saved_model_name = uuid.uuid4().hex[:6].upper() + '.sav'
    pickle.dump(saved_model, open(saved_model_name, 'wb'))
    #Now post this to the persistent Storage
    #Get the URI of the Storage service
    # See if the setting exists if not create it and leave it blank
    if hasattr(conf_settings, 'PERSISTENT_STORAGE'):
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    else:
        response = {"Message" : "Persistent Storage URI has not been configured please configure"}
        status_code = status.HTTP_424_FAILED_DEPENDENCY
        return response, status_code
    #Now do the post
    payload={'file_name': filename,
            'model_type': model_type,
            'num_topics': num_topics,
            'normalisation': normalisation,
            'topic_labels':json.dumps(tfidf_labels)}
    #Open the file
    saved_model = open(saved_model_name,'rb')
    model_image = open(model_name + '.png','rb')
    files={'save_model':saved_model,'topics_image':model_image}
    headers = {}
    save_response = requests.request("POST",persistent_storage + '/storage/',headers=headers,data=payload,files=files)
    #Now Delete the Files
    saved_model.close()
    model_image.close()
    os.remove(saved_model_name)
    os.remove(model_name + '.png')
    os.remove(filename)
    return json.loads(save_response.text), status_code


    
