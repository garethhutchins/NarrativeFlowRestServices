from rest_framework import serializers
from rest_framework.serializers import Serializer, FileField, CharField, IntegerField, ChoiceField

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

class RemoveStopWordSerializer(Serializer):
   post_text = CharField(allow_blank=False)
   stop_word_list = CharField(allow_blank=False) 
   class Meta:
      fields = ['post_text','stop_word_list'] 

class TrainTopicTableSerializer(Serializer):
   file = FileField()
   selected_column = CharField(allow_blank=False)
   model_type = ChoiceField(choices = (("LDA","LDA"),("NMF","NMF")))
   num_topics = IntegerField()
   normalisation = ChoiceField(choices = (("None","None"),("Stemming","Stemming"),("Lemmatisation","Lemmatisation")))
   class Meta:
      fields = ['file','selected_column','model_type','num_topics','normalisation']
