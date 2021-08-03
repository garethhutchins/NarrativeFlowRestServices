#This will remove Stop Words from the provided text
#We will use NLTK but also develop our own aproach
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def list_stop_words(request):
    stops = set(stopwords.words('english'))
    return stops


def remove_stop_words(request):
    text = request._data['text']
    stop_word_list = request._data['stop_word_list']
    if stop_word_list == 'nltk':
        #Now process the text
        #Tokenize the Text

        #Convert to Lower Case
        text = text.lower()

    return 'Test'