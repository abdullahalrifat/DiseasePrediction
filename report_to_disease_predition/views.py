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

from django.http import JsonResponse, HttpResponse
# from chatterbot.trainers import ChatterBotCorpusTrainer


# ACCESS_TOKEN ="EAADbAKIlGVIBALXHqFaTZAPQV3C4KhSJjAlzDmfQnZAeuiTmEOtuvpyHFm8NdmzAmqNdFOlZARm1J98q9JWah9sjCIS1MOqLzKELqWHZA1vtlZBrZCar3Tq1kMSpG9wHbYZBZBZBcdBdKlPP13ZBIbq7XqDOxTrB1g4AQnw8N7Y4LiLNjHCUVd80Os"
ACCESS_TOKEN ="EAADbAKIlGVIBAHVuOoj23vtacyRQUunwULjqzgXHGZCLRKwOsPhU8LKPaR073aBeWoFCZAR8z9yJ71IKSuo4FvXwaXTZBYiZBrZCgtjdXjhryZBoeChA4ExJoQBs64Ds1ILANC0fwH8KZB3hsZAj0HcWsatMaFOCcG3SRxIYiDHm4gYTvArQESfgQJ5ZCXk9kI2ZC02Wp8Y6UVbTePeVlwS4ST"
VERIFY_TOKEN = 'my_voice_is_my_password_verify_me'


def privacy(request):
    return render(request, 'privacy_policy.html', {})


class Predict(views.APIView):

    def get(self, request, version, format=None):
        if not 'verify_token' in request.GET:
            return HttpResponse("Verification token mismatch", status=403)
        token_sent = request.GET['verify_token']
        verifiation = verify_token(token_sent, request)
        if not verifiation:
            return HttpResponse("Verification token mismatch", status=403)
        return HttpResponse("Verification Matched")

    def post(self, request, version, format=None):
        if not 'verify_token' in request.POST:
            return HttpResponse("Verification token mismatch", status=403)
        token_sent = request.POST['verify_token']
        verifiation = verify_token(token_sent, request)
        if verifiation:

            text = request.POST['text']
            #response_data = text.serialize()
            return Response({'text': text}, status=status.HTTP_201_CREATED)
            # serializer = TextSerializer(text, many=True)
        else:
            return HttpResponse("Verification Error")


def verify_token(token_sent, request):
    if token_sent:
        if not token_sent == VERIFY_TOKEN:
            return False
        return True

    return None


def send_message(request,
                 recipient_id,
                 message_text,
                 category="id",
                 message_type="UPDATE"):

    # log("sending message to {recipient}: {text}".format(
        # recipient=recipient_id, text=message_text))

    params = {
        "access_token": ACCESS_TOKEN
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "recipient": {
            category: recipient_id
        },
        "message": {
            "text": message_text["text"]
        },
        "messaging_type": message_type
    })
    r = requests.post("https://graph.facebook.com/v2.6/me/messages",
                      params=params, headers=headers, data=data)
    log(r.text)
    if r.status_code != 200:
        log(r.status_code)


def log(message):  # simple wrapper for logging to stdout on heroku
    print(str(message))
    sys.stdout.flush()
