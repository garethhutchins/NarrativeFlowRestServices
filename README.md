# Rest Services for the Narrative Flow Analyser
Restful services for actions associated to the learning and predicting of the narrative flow of text.
The following Services are supported
## Training Topics from CSV files
This service accepts a CSV file as input.
### POST 
http://server/train_topic_table</br>
{
    'file':'The input csv file',</br>
    'normalisation' : 'None|Stemming|Lemmatisation',</br>
    'model_type' : 'LDA|NMF|TF-IDF',</br>
    'selected_column' : 'Mandatory is more than one column, can be name or index of column',</br>
    'num_topics' : 'The number of topics that is expected. Mandatory for NMF models'</br>
}

