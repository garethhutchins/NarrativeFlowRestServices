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
    'model_type' : 'K-MEANS|LDA|NMF|TF-IDF',<br />
    'selected_column' : 'Mandatory is more than one column, can be name or index of column',<br />
    'num_topics' : 'The number of topics that is expected. Mandatory for k-means & NMF models',<br />
    'label_column' : 'The Labels for TF-IDF Training'<br /> 
}

## Model Management
## GET
Get a Model<br />
Request:<br />
http://server/models/{modelID}<br />
Response:<br>

[<br>
    {<br>
        "name": "7d895514-d1a9-4863-9dc4-4473b51aec15",<br>
        "file_name": "BBC News Train_4NWuC0g.csv",<br>
        "model_type": "K-MEANS",<br>
        "num_topics": 5,<br>
        "normalisation": "None",<br>
        "topic_labels": {<br>
            "1": "Sport",<br>
            "2": "Politics",<br>
            "3": "Technology",<br>
            "4": "Economy",<br>
            "5": "Sport"<br>
        },
        "save_model": "http://127.0.0.1:8001/models/models/7d895514-d1a9-4863-9dc4-4473b51aec15.sav",<br>
        "topics_image": "http://127.0.0.1:8001/models/models/7d895514-d1a9-4863-9dc4-4473b51aec15.png"<br>
    },
    200
]

## PUT
Update a Model<br />
Request:<br />
http://server/models/{modelID}<br />


## Delete<br />
Delete a model<br />
Request:<br />
http://server/models/{modelID}<br />


## Listing Models

### GET
http://server/models/<br />
Get's a list of trained models - example response<br />
[<br />
    {<br />
        "name":\model id",<br />
        "file_name":"Original File name",<br />
        "model_type":"k-means|NMF|LDA|TF-IDF",<br />
        "num_topics":number of topcics,<br />
        "normalisation":"None|Stemming|Lemmatisation",<br />
        "topic_labels":{},<br />
        "save_model":"uri of saved model"<br />
        }<br />
]

## Analysing Text
Analyses the text against a specified model
### POST
http://server/process_text/<br />
{<br />
    "text":"The text to post and analyse",<br />
    "model_id":"The model ID to use for the prediction",<br />
    "window_size":"The size of the window, number of words, or chunk of text to process at a time",<br />
    "window_slide":"The number of words to move along before analysing again"<br />

}

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
