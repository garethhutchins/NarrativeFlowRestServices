from rest_framework import serializers
from rest_framework.serializers import Serializer, FileField

class YourSerializer(serializers.Serializer):
   """Your data serializer, define your fields here."""
   comments = serializers.IntegerField()
   likes = serializers.IntegerField()

class UploadSerializer(Serializer):
   file = FileField()
   class Meta:
      fields = ['file']      