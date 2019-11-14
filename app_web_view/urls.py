from django.conf.urls import url
from django.contrib import admin
from django.urls import path, re_path, include

from app_web_view.views import *

urlpatterns = [
    url(r'^$', home, name='home'),
    url(r'^test', test, name='test'),
    url(r'^diabetes', diabetes, name='diabetes'),
    url(r'^cardio_vascular', cardio_vascular, name='cardio_vascular'),
    url(r'^privacy', privacy, name='privacy'),
]
