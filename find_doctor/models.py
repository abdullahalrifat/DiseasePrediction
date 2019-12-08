from django.db import models
from django.conf import settings

# Create your models here.
from django.utils import timezone


class Doctor(models.Model):
    doctorName = models.CharField(max_length=200)
    doctor_specialist_On = models.CharField(max_length=200)
    doctor_address = models.CharField(max_length=200)
    doctor_phone_no = models.CharField(max_length=200)
    created_date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.doctorName


class Place(models.Model):
    placeName = models.CharField(max_length=200)
    lat = models.DecimalField(max_digits=9, decimal_places=6)
    longi = models.DecimalField(max_digits=9, decimal_places=6)
    created_date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.placeName + " " + str(self.lat) + " " + str(self.longi)


# Create your models here.
class Chamber(models.Model):
    place = models.ForeignKey(Place, on_delete=models.CASCADE)
    doctor = models.ForeignKey(Doctor, on_delete=models.CASCADE)
    created_date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.doctor.doctorName + " " + self.place.placeName
