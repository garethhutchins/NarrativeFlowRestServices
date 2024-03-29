#Configure the Service Settings
from django.core.files.storage import FileSystemStorage
from django.core.files import File
from rest_framework import status
#Save and load the settings
from django.conf import settings as conf_settings
import django


#Used for checking access
import re
import requests
import json

def list_settings(request):
    # See if the setting exists if not create it and leave it blank
    if hasattr(conf_settings, 'PERSISTENT_STORAGE'):
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    else:
        conf_settings.PERSISTENT_STORAGE = ""
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    #TIKA
    if hasattr(conf_settings, 'TIKA'):
        tika = conf_settings.TIKA
    else:
        conf_settings.TIKA = ""
        tika = conf_settings.TIKA
    options = {
        "persistent_storage" : persistent_storage,
        "tika" : tika
    }
    status_code = status.HTTP_200_OK
    return options, status_code

def update_settings(request):
    #Check to see if the uri path is in the request
    if 'persistent_storage' in request.data:
        #Get the URI
        persistent_storage = request._full_data['persistent_storage']
        #Now test to see if it's what is expected
        try:
            get_storage = requests.request("GET",persistent_storage)
        except Exception as e:
            response = {"Message" : "Unable to Reach URI {}".format(e)}
            status_code = status.HTTP_400_BAD_REQUEST
            return response, status_code
        #Check to see if it's ok
        if get_storage.ok != True:
            #It's no ok
            status_code = get_storage.status_code
        #Now check to see if it's what we expect
        try:
            jr = json.loads(get_storage.text)
        except Exception as e:
            response = {"Message" : "Error decoding json {}".format(get_storage.text)}
            status_code = status.HTTP_400_BAD_REQUEST
            return response, status_code
        if 'storage' in jr:
            #We can assume it's ok save the setting
            #Remove the end of the uri though
            s_uri = jr['storage']
            
            conf_settings.PERSISTENT_STORAGE = s_uri.replace("/storage/",'')
            django.setup()
            status_code = status.HTTP_200_OK
            response = {"Message":"Persistent Storage URI Saved"}
            return response, status_code
        else:
            response = {"Message":"Unexptected Response from URI"}
            status_code = status.HTTP_400_BAD_REQUEST
            return response, status_code
        
        

