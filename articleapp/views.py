from django.shortcuts import render,redirect
from .models import User
import mysql.connector
from operator import itemgetter
from django.contrib import messages
from datasets import load_dataset
from .models import Makale
from django.shortcuts import render, get_object_or_404
from django.db.models import Q
import pandas as pd
import fasttext
import fasttext.util
import numpy as np
from .preprocessing import compute_interest_vectors,cosineHesapla,showRecommandations,showSciRecommendation,compute_interest_vectors_forSci




def home(request):
    return render(request, 'home.html')

def login(request):
    if request.method == 'POST':
        email = request.POST['email']
        sifre = request.POST['sifre']

        # Kullanıcıyı veritabanından kontrol et
        try:
            user = User.objects.get(email=email, sifre=sifre)
           
            request.session['user_id'] = user.id
            request.session['email'] = user.email
            # Kullanıcıyı profil sayfasına yönlendir
            return redirect('profile')
        except User.DoesNotExist:
            # Kullanıcı bulunamazsa hata mesajı göster
            messages.error(request, "Geçersiz email veya şifre")
            return redirect('login')

    return render(request, 'login.html')
    
def register(request):
    if request.method == "POST":
        user = User()

        user.ad = request.POST['ad']
        user.soyad = request.POST['soyad']
        user.email = request.POST['email']
        user.sifre= request.POST['sifre']
        user.sifretekrari = request.POST['sifretekrari']
        user.ilgi_alanlari = request.POST['ilgi_alanlari']
        if user.sifre != user.sifretekrari:
            return redirect('register')
        elif user.ad == "" or user.sifre == "":
            messages.info(request,'Bilgiler Eksik')
            return redirect('register')
        else:
            user.save()

    
    return render(request, 'register.html')


#Bu fonksiyon aracılığıyla makale datasetini veritabanına aktarıyoruz
def load_and_save_dataset(request):
 
    dataset = load_dataset("memray/inspec") #Hugging Face'den alınan veri seti

    for key in dataset.keys():
        for makale in dataset[key]:
            #Her bir makaleyi veri setinden okuyoruz
            makale_bilgisi = Makale(
                name=makale['name'],
                title=makale['title'],
                abstract=makale['abstract'],
                fulltext=makale['fulltext'],
                keywords=makale['keywords']
            )
            #Veritabanına kaydediyoruz
            makale_bilgisi.save()

    return render(request, 'dataset_loaded.html')


def profile(request):
    if request.method == 'POST':
        user = User.objects.get(email=request.session['email'])
        user.ilgi_alanlari = request.POST.get('ilgi_alanlari')
        user.save()


    user = User.objects.get(email=request.session['email'])
    ad_soyad = f"{user.ad} {user.soyad}"
    userId = user.id
    ilgi_alanlari = user.ilgi_alanlari.split(',')

    query = request.GET.get('q')
    benzer_makaleler = Makale.objects.none()

    if query:
        # Anahtar kelimeye göre makaleleri filtrele
        makaleler = Makale.objects.filter(
            Q(keywords__icontains=query)
        )
        benzer_makaleler = makaleler.distinct()  # Çift girişleri önlemek için distinct() kullanın

    else:
        # İlgi alanlarına göre makaleleri filtrele
        for ilgi in ilgi_alanlari:
            makaleler = Makale.objects.filter(abstract__icontains=ilgi)
            benzer_makaleler = benzer_makaleler.union(makaleler)

    

    compute_interest_vectors(userId,ilgi_alanlari)
    compute_interest_vectors_forSci(userId,ilgi_alanlari)
    cosineHesapla()
    article_names = showRecommandations()
    sciArticles = showSciRecommendation()

    recommended_articles = Makale.objects.filter(name__in=article_names)
    recommended_scibert = Makale.objects.filter(name__in = sciArticles)

    context = {'benzer_makaleler': benzer_makaleler, 'ad_soyad': ad_soyad, 'recommended_articles' : recommended_articles,'recommended_scibert': recommended_scibert}

    return render(request, 'profile.html', context)


def makale_detay(request, makale_id):
    # Makaleyi veritabanından alıyoruz
    makale = get_object_or_404(Makale, pk=makale_id)

    context = {'makale': makale}
    return render(request, 'makale_detay.html', context)
   
   

