#This will update the topic labels of a model
from rest_framework import response
from django.conf import settings as conf_settings
from rest_framework import status
import requests
import json

def list_models():
    # See if the setting exists if not create it and leave it blank
    if hasattr(conf_settings, 'PERSISTENT_STORAGE'):
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    else:
        response = {"Message" : "Persistent Storage URI has not been configured please configure"}
        status_code = status.HTTP_424_FAILED_DEPENDENCY
        return response, status_code
    persistent_storage_models = persistent_storage + "/storage/"
    models = requests.get(persistent_storage_models)
    models_json = json.loads(models.content)
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
        #Now sent the update to the storage Service
        models_json['topic_labels'] = topic_labels
        headers = {}
        #Need to get the model file down and then submit the model with updated image
        update_response = requests.request("PUT",persistent_storage + '/storage/' + id,headers=headers,data=models_json)#,files=files) 
    return response