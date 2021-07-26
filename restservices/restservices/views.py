from django.shortcuts import render
from rest_framework import views
from rest_framework.response import Response

from .serializers import YourSerializer

class YourView(views.APIView):

    def get(self, request):
        yourdata= [{"likes": 10, "comments": 0}, {"likes": 4, "comments": 23}]
        results = YourSerializer(yourdata, many=True).data
        return Response(results)

