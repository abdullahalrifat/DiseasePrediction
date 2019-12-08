from django.conf.urls import url
from django.contrib import admin
from django.urls import path, re_path, include

from find_doctor.views import *

urlpatterns = [

    url(r'^search$', Search.as_view(), name='search'),

]
