from django.shortcuts import render
from rest_framework import views
from rest_framework import response
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet

from .serializers import YourSerializer
from rest_framework.decorators import parser_classes

from rest_framework.parsers import FileUploadParser
from rest_framework import status

from .serializers import GetTextSerializer, RemoveStopWordSerializer, TrainTopicTableSerializer, ServiceSettingsSerialiser, ModelSerializer, ProcessTextSerializer


from .get_text import get_text
from .stop_words import remove_stop_words, list_stop_words
from .train_topic_table import list_train_table_options, train_table
from .service_settings import list_settings, update_settings
from .update_model import list_models, update_labels, get_model
from .process_text import list_train_table_options, predict_text

#Get the Text Elements from Files
class GetTextViewSet(ViewSet):
    #Define the Get Text Serializer
    serializer_class = GetTextSerializer

    def list(self, request):
        options = {
            'file' : 'file to post',
            'tika' : 'Boolean if tika is required',
            'selected_column':'The column if a csv file is used'
        }
        return Response(options)

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
        response= list_train_table_options(request)
        return Response(response, status=status.HTTP_200_OK)
    #Do a post of content
    def create(self, request):
        response, status_code = train_table(request)
        return Response(response, status = status_code)

#Manage Models
class ListModelsViewSet(ViewSet):
    
    def list(self,request):
        response = list_models()
        
        return Response(response, status=status.HTTP_200_OK)   
                

class GetModelViewSet(ViewSet):
    serializer_class = ModelSerializer
    def list(self,request):
        response = get_model(request)
        
        return Response(response, status=status.HTTP_200_OK)  
    #Update the labels
    def put(self,request):
        response, status_code = update_labels(request)
        return Response(response, status_code)

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

#Process Document
class ProcessTextViewSet(ViewSet):
    serializer_class = ProcessTextSerializer
    #List the Settings
    def list(self,request):
        response, status_code = list_train_table_options(request)
        return Response(response,status_code)
    def create(self,request):
        response, status_code = predict_text(request)
        return Response(response,status_code)
        
