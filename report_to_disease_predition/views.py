from django.shortcuts import render

# Create your views here.
from rest_framework import generics, views
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework import views, status


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


# ACCESS_TOKEN ="EAADbAKIlGVIBALXHqFaTZAPQV3C4KhSJjAlzDmfQnZAeuiTmEOtuvpyHFm8NdmzAmqNdFOlZARm1J98q9JWah9sjCIS1MOqLzKELqWHZA1vtlZBrZCar3Tq1kMSpG9wHbYZBZBZBcdBdKlPP13ZBIbq7XqDOxTrB1g4AQnw8N7Y4LiLNjHCUVd80Os"
ACCESS_TOKEN ="EAADbAKIlGVIBAHVuOoj23vtacyRQUunwULjqzgXHGZCLRKwOsPhU8LKPaR073aBeWoFCZAR8z9yJ71IKSuo4FvXwaXTZBYiZBrZCgtjdXjhryZBoeChA4ExJoQBs64Ds1ILANC0fwH8KZB3hsZAj0HcWsatMaFOCcG3SRxIYiDHm4gYTvArQESfgQJ5ZCXk9kI2ZC02Wp8Y6UVbTePeVlwS4ST"
VERIFY_TOKEN = 'my_voice_is_my_password_verify_me'


# building diabetes dataset from pima
def build_diabetes_pima():

    diab = pd.read_csv(staticfiles_storage.open('dataset/diabetes.csv'))

    diab2 = diab[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Outcome']]
    features = diab2[diab2.columns[:4]]

    train1, test1 = train_test_split(diab2, test_size=0.25, random_state=0)

    train_X1 = train1[train1.columns[:4]]
    test_X1 = test1[test1.columns[:4]]
    train_Y1 = train1['Outcome']
    test_Y1 = test1['Outcome']

    # print(train_Y1.head(5))
    model = LogisticRegression()
    model.fit(train_X1, train_Y1)
    prediction = model.predict(test_X1)

    # print('The accuracy of the Model : ', metrics.accuracy_score(prediction, test_Y1))
    # print('\nConfusion Matrix - \n')
    # print(confusion_matrix(prediction, test_Y1))
    # print('\n\n')
    return model


# creating istance of pima database model
diabetes_model = build_diabetes_pima()


# training the dataset
def train(request):
    return None


# predict from the user given data
class Predict(views.APIView):

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
            if type == "diabetes_pima":
                glucose = request.POST['glucose']
                bmi = request.POST['bmi']
                age = request.POST['age']
                diabetes_pedigree_function = request.POST['diabetes_pedigree_function']
                prediction = diabetes_model.predict(np.array([[int(glucose), int(bmi), int(age), float(diabetes_pedigree_function)]]))
                return Response({'prediction': prediction}, status=status.HTTP_201_CREATED)
            else:
                return Response({'status': "No Model Selected"}, status=status.HTTP_201_CREATED)
            # serializer = TextSerializer(text, many=True)
        else:
            return Response("Verification Error")


# verfiying user from token
def verify_token(token_sent, request):
    if token_sent:
        if not token_sent == VERIFY_TOKEN:
            return False
        return True

    return None
