from rest_framework import serializers
from rest_framework.serializers import Serializer, FileField, CharField

class YourSerializer(serializers.Serializer):
   """Your data serializer, define your fields here."""
   comments = serializers.IntegerField()
   likes = serializers.IntegerField()

class GetTextSerializer(Serializer):
   file = FileField()
   tika = CharField(allow_blank=True)
   selected_column = CharField(allow_blank=True)
   class Meta:
      fields = ['file','tika','selected_column']      