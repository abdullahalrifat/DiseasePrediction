from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Create your views here.


# home page
def home(request):
    return render(request, 'index.html', {})


# priacy page
def privacy(request):
    return render(request, 'privacy_policy.html', {})