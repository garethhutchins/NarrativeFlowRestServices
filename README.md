# Rest Services for the Narrative Flow Analyser
Restful services for actions associated to the learning and predicting of the narrative flow of text.
The following Services are supported
## Training Topics from CSV files
This service accepts a CSV file as input.<br />
http://server/train_topic_table<br />
### POST 
Create a new model from a CSV file

{
    'file':'The input csv file',<br />
    'normalisation' : 'None|Stemming|Lemmatisation',<br />
    'model_type' : 'LDA|NMF|TF-IDF',<br />
    'selected_column' : 'Mandatory is more than one column, can be name or index of column',<br />
    'num_topics' : 'The number of topics that is expected. Mandatory for NMF models'<br />
}

### GET
Get's a list of trained models - example response<br />
[<br />
    {<br />
        "name":\model id",<br />
        "file_name":"Original File name",<br />
        "model_type":"NMF|LDA|TF-IDF",<br />
        "num_topics":number of topcics,<br />
        "normalisation":"None|Stemming|Lemmatisation",<br />
        "topic_labels":{},<br />
        "save_model":"uri of saved model"<br />
        }<br />
]

## Getting text from documents
This service gets the text from text searchable documents or CSV files.<br />
http:/server/gettext/
### POST
{<br />
    "file": "file to post",<br />
    "tika": "Boolean if tika is required",<br />
    "selected_column": "The column if a csv file is used"<br />
}
## GET
Gets a description of the service

## Get the list of Stopwords used
http://server/stopwords/
