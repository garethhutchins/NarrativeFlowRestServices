from django.shortcuts import render
from rest_framework import views
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet

from .serializers import YourSerializer
from rest_framework.decorators import parser_classes

from rest_framework.parsers import FileUploadParser
from rest_framework import status

from .serializers import GetTextSerializer, RemoveStopWordSerializer, TrainTopicTableSerializer


from .get_text import get_text
from .stop_words import remove_stop_words, list_stop_words
from .train_topic_table import list_options, train_table

class YourView(views.APIView):

    def get(self, request):
        yourdata= [{"likes": 10, "comments": 0}, {"likes": 4, "comments": 23}]
        results = YourSerializer(yourdata, many=True).data
        return Response(results)


#Get the Text Elements from Files
class GetTextViewSet(ViewSet):
    #Define the Get Text Serializer
    serializer_class = GetTextSerializer

    def list(self, request):
        return Response("GET API")

    def create(self, request):
        response = get_text(request)
        
        return Response(response)

#Remove the Stop Words
class StopWordsViewSet(ViewSet):
    serializer_class = RemoveStopWordSerializer
    #list the Stop words that we'll use
    def list(self, request):
        response = list_stop_words(request)
        return Response(response)

    def create(self, request):
        response = remove_stop_words(request)
        return Response(response)

#Train Topics from Table
class TrainTopicTableViewSet(ViewSet):
    serializer_class = TrainTopicTableSerializer
    #List the Parameters
    def list(self,request):
        response = list_options(request)
        return Response(response)
    #Do a post of content
    def create(self, request):
        response = train_table(request)
        return Response(response)