from django.shortcuts import render
from rest_framework import views
from rest_framework.response import Response
from rest_framework.viewsets import ViewSet

from .serializers import YourSerializer
from rest_framework.decorators import parser_classes

from rest_framework.parsers import FileUploadParser
from rest_framework import status

from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from .serializers import UploadSerializer

class YourView(views.APIView):

    def get(self, request):
        yourdata= [{"likes": 10, "comments": 0}, {"likes": 4, "comments": 23}]
        results = YourSerializer(yourdata, many=True).data
        return Response(results)



# ViewSets define the view behavior.
class UploadViewSet(ViewSet):
    serializer_class = UploadSerializer

    def list(self, request):
        return Response("GET API")

    def create(self, request):
        
        file_uploaded = request.FILES.get('file')
        content_type = file_uploaded.content_type
        response = "POST API and you have uploaded a {} file".format(content_type)
        return Response(response)
