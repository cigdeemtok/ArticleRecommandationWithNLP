#Kullanıcıları admin paneli üzerinden kontrol ediyoruz
from django.contrib import admin

from .models import User
admin.site.register(User)

