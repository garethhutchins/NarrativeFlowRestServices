"""restservices URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path, re_path
from restservices.restservices import views
from rest_framework import routers
from restservices.restservices.views import GetTextViewSet, StopWordsViewSet, TrainTopicTableViewSet, ServiceSettingsViewSet, ListModelsViewSet, GetModelViewSet, ProcessTextViewSet
from django.conf.urls import url

router = routers.DefaultRouter()
router.register(r'gettext', GetTextViewSet, basename="gettext")
router.register(r'stopwords', StopWordsViewSet, basename="stopwords")
router.register(r'train_topic_table',TrainTopicTableViewSet, basename='train_topic_table')
router.register(r'service_settings',ServiceSettingsViewSet, basename='service_settings')
router.register(r'models',ListModelsViewSet, basename='list_models')
router.register(r"models/[a-z0-9\-]{36}",GetModelViewSet, basename='get_model')
router.register(r'process_text',ProcessTextViewSet, basename='process_text')
urlpatterns = [
    #path('', views.YourView.as_view()),
    path('',include(router.urls)),
    
]
