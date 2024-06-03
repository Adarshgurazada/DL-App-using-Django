from django.urls import path
from . import views

urlpatterns = [
    path('dl_model/', views.dl_model_view, name='dl_model'),
]
