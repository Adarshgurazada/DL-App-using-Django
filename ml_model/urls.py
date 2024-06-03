from django.urls import path
from . import views

urlpatterns = [
    path('ml_model/', views.ml_model_view, name='ml_model'),  # URL pattern for ML model view
]
