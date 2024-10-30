from django.db import models

#Veritabanı için User modeli oluşturduk
class User(models.Model):
    id = models.AutoField(primary_key=True)
    ad = models.CharField(max_length=30)
    soyad = models.CharField(max_length=30)
    email = models.CharField(max_length=30)
    sifre = models.CharField(max_length=30)
    sifretekrari= models.CharField(max_length=30)
    ilgi_alanlari = models.CharField(max_length=30)


    
#Veritabanı için Dataset modeli oluşturduk
class Dataset(models.Model):
    name = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    abstract = models.TextField()
    fulltext = models.TextField()
    keywords = models.TextField()


class Makale(models.Model):
    name = models.CharField(max_length=255)
    title = models.CharField(max_length=255)
    abstract = models.TextField()
    fulltext = models.TextField()
    keywords = models.TextField()