from django.contrib import admin
from django.urls import path
from .views import predict_disease, plant_model

urlpatterns = [
    path('predict/', predict_disease, name="predict"),
    path('home/', plant_model, name="home"),
]