from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Create your views here.


# home page
def home(request):
    return render(request, 'index.html', {})


# home page
def test(request):
    return render(request, 'test.html', {})


# home page
def diabetes(request):
    return render(request, 'diabetes.html', {})


# home page
def cardio_vascular(request):
    return render(request, 'cardio_vascular.html', {})

# priacy page
def privacy(request):
    return render(request, 'privacy_policy.html', {})