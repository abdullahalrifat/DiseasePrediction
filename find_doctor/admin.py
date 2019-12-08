from django.contrib import admin

# Register your models here.
from find_doctor.models import *

admin.site.register(Doctor)
admin.site.register(Place)
admin.site.register(Chamber)
