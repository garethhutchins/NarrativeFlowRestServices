#This will remove Stop Words from the provided text
#We will use NLTK but also develop our own aproach
import nltk
import requests
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def list_stop_words(request):
    stop_word_list = request.GET.get('stop_word_list','')
    if stop_word_list == '':
        stops = set(stopwords.words('english'))
    if stop_word_list == 'nltk':
        stops = set(stopwords.words('english'))
    return stops


def remove_stop_words(request):
    temp_request_data = request.data.copy()
    post_text = ""
    if 'post_text' in temp_request_data:
        post_text = temp_request_data['post_text']
        post_stop_word_list = temp_request_data['stop_word_list']
        if post_stop_word_list == 'nltk':
            #Now process the text
            #Convert to Lower Case
            post_text = post_text.lower()
            #Tokenize the Text

            

    return post_text