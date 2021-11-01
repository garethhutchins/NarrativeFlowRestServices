#Configure the Service Settings
from django.core.files.storage import FileSystemStorage
from django.core.files import File
from rest_framework import status
#Save and load the settings
from django.conf import settings as conf_settings

from smb.SMBConnection import SMBConnection

#Used for checking access
import re

def list_settings(request):
    # See if the setting exists if not create it and leave it blank
    if hasattr(conf_settings, 'PERSISTENT_STORAGE'):
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    else:
        conf_settings.PERSISTENT_STORAGE = ""
        persistent_storage = conf_settings.PERSISTENT_STORAGE
    options = {
        "persistent_storage" : persistent_storage
    }
    status_code = status.HTTP_200_OK
    return options, status_code

def update_settings(request):
    #Check to see if the uri path is in the request
    if 'persistent_storage' in request.data:
        #Now see if you can write to that directory
        persistent_storage = request._full_data['persistent_storage']
        #Create another container to host the persistent storage
        
        

