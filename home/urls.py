from django.urls import path
from . import views

urlpatterns = [
    path('api/find-complementary-teammates', views.find_complementary_teammates),  # No trailing slash
    path('api/student-profile/<str:uid>/', views.get_student_profile),
]