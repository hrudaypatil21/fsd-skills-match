from django.db import models
from django.contrib.auth.models import User

class StudentProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    contact = models.CharField(max_length=100)
    domain = models.CharField(max_length=100, blank=True, null=True)
    role_preference = models.CharField(max_length=100, blank=True, null=True)
    profile_picture = models.URLField(blank=True, null=True)
    projects = models.JSONField(default=list)
    linkedin = models.URLField(blank=True, null=True)
    github = models.URLField(blank=True, null=True)
    portfolio = models.URLField(blank=True, null=True)
    certifications = models.JSONField(default=list)
    is_public = models.BooleanField(default=True)
    experience = models.JSONField(default=list)
    bio = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.name

class UserSkill(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    level = models.CharField(max_length=50)
    
    def __str__(self):
        return f"{self.user.username} - {self.name} ({self.level})"

class CachedEmbedding(models.Model):
    text = models.TextField(unique=True)
    embedding = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['text']),
        ]