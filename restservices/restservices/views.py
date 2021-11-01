from django.shortcuts import render
from rest_framework import views
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet

from .serializers import YourSerializer
from rest_framework.decorators import parser_classes

from rest_framework.parsers import FileUploadParser
from rest_framework import status

from .serializers import GetTextSerializer, RemoveStopWordSerializer, TrainTopicTableSerializer, ServiceSettingsSerialiser


from .get_text import get_text
from .stop_words import remove_stop_words, list_stop_words
from .train_topic_table import list_options, train_table
from .service_settings import list_settings, update_settings

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
        response= list_options(request)
        return Response(response, status=status.HTTP_200_OK)
    #Do a post of content
    def create(self, request):
        response, status_code = train_table(request)
        return Response(response, status = status_code)

#Settings
class ServiceSettingsViewSet(ViewSet):
    serializer_class = ServiceSettingsSerialiser
    #List the Settings
    def list(self,request):
        response, status_code = list_settings(request)
        return Response(response, status_code)
    #Update the Settings
    def put(self,request):
        response, status_code = update_settings(request)
        return Response(response, status_code)