from django.urls import path
from . import views

urlpatterns = [
    path('api/find-complementary-teammates/', views.find_complementary_teammates, name='find_complementary'),
]