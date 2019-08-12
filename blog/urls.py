from django.contrib import admin
from django.urls import path
from .views import view_index, view_single, view_list

urlpatterns = [
    path('', view_index, name='home'),
    path('post/<slug:slug>/', view_single, name='view_single'),
    path('list/<slug:category_slug>/', view_list, name='view_list'),
]
