#Get Text Function from posted Files
import requests
import re
import pandas as pd
import json
from django.core.files.storage import FileSystemStorage

def get_text(request):
    #Get the variables from the request
        #Get the File Object
        file_uploaded = request.FILES.get('file')
        #Content Type
        content_type = file_uploaded.content_type
        #File Name
        name = file_uploaded.name
        #Save the file so it can be read
        fs = FileSystemStorage()
        filename = fs.save(name, file_uploaded)
        full_file_path = fs.location + "/" + filename
        
        #Now see if we need to send it to Tika
        if 'tika' in request._data:
            tika_url = request._data['tika']
            if tika_url != '':
                #open the file for reading
                with open(full_file_path, 'rb') as f:
                    payload=f.read()
                #Add the header
                headers = {
                    'Content-Type': content_type
                    }
                #Now delete the file
                fs.delete(filename)
                tika_response = requests.request("PUT", tika_url, headers=headers, data=payload)
                response = tika_response.text
                if tika_response.status_code != 200:
                    response = tika_response.reason
        #See if it's a table file
        if 'selected_column' in request._data:
            selected_column = request._data['selected_column']
            if selected_column != '':
                #Load the files as a pandas dataframe
                df = pd.read_csv(full_file_path,index_col=0)
                #Now delete the file
                fs.delete(filename)
                #Now only keep the selected column
                df = df[selected_column]
                
                json_records = df.to_json(orient='records')
                data = []
                data = json.loads(json_records)
                response = data
        return response