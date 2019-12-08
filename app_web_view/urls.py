from django.conf.urls import url
from django.contrib import admin
from django.urls import path, re_path, include

from app_web_view.views import *

urlpatterns = [
    url(r'^$', home, name='home'),
    url(r'^test', test, name='test'),
    url(r'^doctor_find', doctor_find, name='doctor_find'),
    url(r'^diabetes$', diabetes, name='diabetes'),
    url(r'^diabetes_medicine', diabetes_medicine, name='diabetes_medicine'),
    url(r'^diabetes_diet_list', diabetes_diet_list, name='diabetes_diet_list'),
    url(r'^diabetes_exercise', diabetes_exercise, name='diabetes_exercise'),
    url(r'^cardio_vascular$', cardio_vascular, name='cardio_vascular'),
    url(r'^cardio_vascular_medicine', cardio_vascular_medicine, name='cardio_vascular_medicine'),
    url(r'^cardio_vascular_diet_list', cardio_vascular_diet_list, name='cardio_vascular_diet_list'),
    url(r'^cardio_vascular_exercise', cardio_vascular_exercise, name='cardio_vascular_exercise'),
    url(r'^privacy', privacy, name='privacy'),
]
