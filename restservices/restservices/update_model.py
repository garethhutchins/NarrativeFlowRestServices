#This will update the topic labels of a model
import pickle
from rest_framework import response
from django.conf import settings as conf_settings
from rest_framework import status
import requests
import json
import urllib.request
import os

from .common_processing import plot_top_words, plot_kmeans


def list_models():
    # See if the setting exists if not create it and leave it blank
    if hasattr(conf_settings, 'PERSISTENT_STORAGE'):
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    else:
        response = {"Message" : "Persistent Storage URI has not been configured please configure"}
        status_code = status.HTTP_424_FAILED_DEPENDENCY
        return response, status_code
    persistent_storage_models = persistent_storage + "/storage/"
    try:
        models = requests.get(persistent_storage_models)
    except Exception as e:
        response = {'Message':str(e),'URI':persistent_storage_models}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    try:
        models_json = json.loads(models.content)
    except:
        response = {'Message':models.content,'URI':persistent_storage_models}
        status_code = status.HTTP_400_BAD_REQUEST
        return response, status_code
    return models_json

def get_model(request):
    # See if the setting exists if not create it and leave it blank
    if hasattr(conf_settings, 'PERSISTENT_STORAGE'):
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    else:
        response = {"Message" : "Persistent Storage URI has not been configured please configure"}
        status_code = status.HTTP_424_FAILED_DEPENDENCY
        return response, status_code
    id = request._request.path
    id = id.replace("/models/","")
    persistent_storage_models = persistent_storage + "/storage/" + id
    models = requests.get(persistent_storage_models)
    models_json = json.loads(models.content)
    response = models_json
    return response, models.status_code

def delete_model(request):
    # See if the setting exists if not create it and leave it blank
    if hasattr(conf_settings, 'PERSISTENT_STORAGE'):
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    else:
        response = {"Message" : "Persistent Storage URI has not been configured please configure"}
        status_code = status.HTTP_424_FAILED_DEPENDENCY
        return response, status_code
    id = request._request.path
    id = id.replace("/models/","")
    persistent_storage_models = persistent_storage + "/storage/" + id
    models = requests.delete(persistent_storage_models)
    if models.status_code == '204':
        response = {'Message':'Model Deleted'}
    else:
        response = {'Message':models.text}
    return response, models.status_code

def update_labels(request):
    #Get the information from the model ID
    # See if the setting exists if not create it and leave it blank
    if hasattr(conf_settings, 'PERSISTENT_STORAGE'):
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    else:
        response = {"Message" : "Persistent Storage URI has not been configured please configure"}
        status_code = status.HTTP_424_FAILED_DEPENDENCY
        return response, status_code
    #Get the Model ID from the request
    id = request._request.path
    id = id.replace("/models/","")
    #Get the json response from the model
    persistent_storage_models = persistent_storage + "/storage/" + id
    models = requests.get(persistent_storage_models)
    models_json = json.loads(models.content)
    if 'topic_labels' in request.data:
        topic_labels = request._full_data['topic_labels']
        #Now see if it's valid json
        try:
            json_topics = json.loads(topic_labels)
        except Exception as e:
            response = {"Message":"Invalid Json Format {}".format(e)}
            return response, status.HTTP_400_BAD_REQUEST
        
        #Get the files from the storage service
        saved_model = pickle.load(urllib.request.urlopen(models_json['save_model']))
        
        #Now send the update to the storage Service
        models_json['topic_labels'] = topic_labels
        if models_json['model_type'] == 'K-MEANS':
            #We need to do something different for k-Means
            plot_image = plot_kmeans(list(json_topics.values()),saved_model['vectorizer'])
        else:    
            feature_names=saved_model['vectorizer'].get_feature_names()
            plot_image = plot_top_words(saved_model['model'], feature_names, models_json['num_topics'], 'Topics in NMF model',models_json['num_topics'],json_topics)
        #Save the plot
        plot_image.savefig(models_json['name'] + '.png')
        #Save the model again so it can be updated
        save_file = open(models_json['name']+'.sav', 'wb')
        pickle.dump(saved_model,save_file)
        save_file.close()
        headers = {}
        save_model = open(models_json['name']+'.sav','rb')
        save_image = open(models_json['name'] + '.png','rb')
        files={'save_model':save_model,'topics_image':save_image}
        #Need to get the model file down and then submit the model with updated image
        update_response = requests.request("PUT",persistent_storage + '/storage/' + id,headers=headers,data=models_json,files=files) 
        #Delete the temp files too
        save_model.close()
        save_image.close()
        os.remove(models_json['name'] + '.png')
        os.remove(models_json['name'] + '.sav')
    return json.loads(update_response.text), update_response.status_code