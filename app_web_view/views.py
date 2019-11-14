from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

verify_token = "my_voice_is_my_password_verify_me"


# home page
def home(request):
    return render(request, 'index.html', {})


# home page
def test(request):
    return render(request, 'test.html', {})


# home page
def diabetes(request):
    return render(request, 'diabetes.html', {"verify_token": verify_token})


# home page
def cardio_vascular(request):
    return render(request, 'cardio_vascular.html', {"verify_token": verify_token})

# priacy page
def privacy(request):
    return render(request, 'privacy_policy.html', {})