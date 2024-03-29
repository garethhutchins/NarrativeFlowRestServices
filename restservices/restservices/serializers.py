from rest_framework import serializers
from rest_framework import fields
from rest_framework.fields import BooleanField
from rest_framework.serializers import Serializer, FileField, CharField, IntegerField, ChoiceField, URLField, JSONField

class YourSerializer(serializers.Serializer):
   """Your data serializer, define your fields here."""
   comments = serializers.IntegerField()
   likes = serializers.IntegerField()

class GetTextSerializer(Serializer):
   file = FileField()
   tika = BooleanField(default=True)
   selected_column = CharField(allow_blank=True)
   class Meta:
      fields = ['file','tika','selected_column']   

class RemoveStopWordSerializer(Serializer):
   post_text = CharField(allow_blank=False)
   stop_word_list = ChoiceField(choices = ("nltk","nltk"))
   class Meta:
      fields = ['post_text','stop_word_list'] 

class TrainTopicTableSerializer(Serializer):
   file = FileField()
   selected_column = CharField(allow_blank=False)
   model_type = ChoiceField(choices = (("K-MEANS","K-MEANS"),("LDA","LDA"),("NMF","NMF"),("TF-IDF","TF-IDF")))
   num_topics = IntegerField()
   normalisation = ChoiceField(choices = (("None","None"),("Stemming","Stemming"),("Lemmatisation","Lemmatisation")))
   label_column = CharField()
   class Meta:
      fields = ['file','selected_column','model_type','num_topics','normalisation','label_column']

class ModelSerializer(Serializer):
   topic_labels = JSONField()
   class Meta:
      fields = ['topic_labels']

class ServiceSettingsSerialiser(Serializer):
   persistent_storage = URLField()
   tika = URLField()
   class Meta:
      fields = ['persistent_storage,tika']

class ProcessTextSerializer(Serializer):
   text = CharField(allow_blank=False)
   window_size = IntegerField()
   window_slide = IntegerField()
   model_id = CharField()
   class Meta:
      fields = ['text','window_size','window_slide','model_id']
