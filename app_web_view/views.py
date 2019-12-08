from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

verify_token = "my_voice_is_my_password_verify_me"


# home page
def home(request):
    return render(request, 'index.html', {})


def doctor_find(request):
    return render(request, 'doctor_find.html', {"verify_token": verify_token})


# home page
def test(request):
    return render(request, 'test.html', {})


# home page
def diabetes_medicine(request):
    return render(request, 'diabetes_medicine.html', {})


# home page
def diabetes_diet_list(request):
    return render(request, 'diabetes_diet_list.html', {})


# home page
def diabetes_exercise(request):
    return render(request, 'diabetes_exercise.html', {})


# home page
def cardio_vascular_medicine(request):
    return render(request, 'cardio_vascular_medicine.html', {})


# home page
def cardio_vascular_diet_list(request):
    return render(request, 'cardio_vascular_diet_list.html', {})


# home page
def cardio_vascular_exercise(request):
    return render(request, 'cardio_vascular_exercise.html', {})


# home page
def diabetes(request):
    return render(request, 'diabetes.html', {"verify_token": verify_token})


# home page
def cardio_vascular(request):
    return render(request, 'cardio_vascular.html', {"verify_token": verify_token})

# priacy page
def privacy(request):
    return render(request, 'privacy_policy.html', {})