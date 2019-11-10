from django.conf.urls import url
from django.contrib import admin
from django.urls import path, re_path, include

from app_web_view.views import *

urlpatterns = [
    # url('', home, name='home'),
    url(r'^privacy', privacy, name='privacy'),
]
