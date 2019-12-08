from django.shortcuts import render

# Create your views here.
from rest_framework import generics, views
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework import views, status

from find_doctor.models import Doctor, Place, Chamber

from report_to_disease_predition.serializers import TextSerializer
import json
import sys
import warnings
import nltk
import numpy as np
import random
import string # to process standard python strings
import requests

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier

from sklearn import metrics
import io

import itertools
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import io

import itertools
import os
import pickle
from django.contrib.staticfiles.templatetags.staticfiles import static
from django.contrib.staticfiles.storage import staticfiles_storage
from sklearn.linear_model import LogisticRegression
from django.utils.decorators import method_decorator
from rest_framework.permissions import IsAuthenticated  # <-- Here

from django.http import JsonResponse, HttpResponse
# from chatterbot.trainers import ChatterBotCorpusTrainer
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework.authentication import SessionAuthentication, BasicAuthentication

from django.db import connection
from django.core import serializers


# ACCESS_TOKEN ="EAADbAKIlGVIBALXHqFaTZAPQV3C4KhSJjAlzDmfQnZAeuiTmEOtuvpyHFm8NdmzAmqNdFOlZARm1J98q9JWah9sjCIS1MOqLzKELqWHZA1vtlZBrZCar3Tq1kMSpG9wHbYZBZBZBcdBdKlPP13ZBIbq7XqDOxTrB1g4AQnw8N7Y4LiLNjHCUVd80Os"
ACCESS_TOKEN ="EAADbAKIlGVIBAHVuOoj23vtacyRQUunwULjqzgXHGZCLRKwOsPhU8LKPaR073aBeWoFCZAR8z9yJ71IKSuo4FvXwaXTZBYiZBrZCgtjdXjhryZBoeChA4ExJoQBs64Ds1ILANC0fwH8KZB3hsZAj0HcWsatMaFOCcG3SRxIYiDHm4gYTvArQESfgQJ5ZCXk9kI2ZC02Wp8Y6UVbTePeVlwS4ST"
VERIFY_TOKEN = 'my_voice_is_my_password_verify_me'


def search_doctor_from_location_and_speciality(request):
    place = request.POST['place']
    speciality = request.POST['speciality']
    cursor = connection.cursor()

    cursor.execute('''Select * from (SELECT t2.placeName, find_doctor_doctor.doctorName, find_doctor_doctor.doctor_address, find_doctor_doctor.doctor_specialist_On, find_doctor_doctor.doctor_phone_no , SQRT(Power(t1.longi - t2.longi, 2) 
+ Power(t1.lat - t2.lat, 2)) as distance 
FROM find_doctor_place as t1 JOIN find_doctor_place as t2 join find_doctor_doctor join find_doctor_chamber
WHERE t1.placeName = %s and find_doctor_doctor.doctor_specialist_On = %s and find_doctor_doctor.id=find_doctor_chamber.doctor_id and t2.id=find_doctor_chamber.place_id) as t3 order by t3.distance''', [place, speciality])

#     cursor.execute('''SELECT t2.placeName, SQRT(Power(t1.longi - t2.longi, 2)
# + Power(t1.lat - t2.lat, 2)) as distance
# FROM find_doctor_place as t1 JOIN find_doctor_place as t2
# WHERE t1.placeName = %s order by distance''', [place])

    rows = cursor.fetchall()
    # doctors = Doctor.objects.filter(doctor_specialist_On=speciality)
    # place = Place.objects.filter(placeName=place)
    # chamber = Chamber.objects.filter(doctor=doctors, place=place)
    print(rows)

    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


# predict from the user given data
class Search(views.APIView):

    # user get request serve here
    def get(self, request, version, format=None):
        if not 'verify_token' in request.GET:
            return Response("Verification token mismatch", status=403)

        token_sent = request.GET['verify_token']
        verifiation = verify_token(token_sent, request)
        if not verifiation:
            return Response("Verification token mismatch", status=403)
        return Response("Verification Matched")

    # user post request serve here
    def post(self, request, version, format=None):
        if not 'verify_token' in request.POST:
            return HttpResponse("Verification token mismatch", status=403)

        token_sent = request.POST['verify_token']
        verifiation = verify_token(token_sent, request)
        if verifiation:

            type = request.POST['type']
            print(type)
            if type == "search_by_speciality_and_area":
                response = search_doctor_from_location_and_speciality(request)
                return Response({'response': response}, status=status.HTTP_201_CREATED)
            else:
                return Response({'status': "No Model Selected"}, status=status.HTTP_201_CREATED)
            # serializer = TextSerializer(text, many=True)
            # return Response("Verification Done")
        else:
            return Response("Verification Error")


# verfiying user from token
def verify_token(token_sent, request):
    if token_sent:
        if not token_sent == VERIFY_TOKEN:
            return False
        return True

    return None
from django.shortcuts import render

# Create your views here.
