# serializers.py
from rest_framework import serializers
from .models import StudentProfile, UserSkill

class UserSkillSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserSkill
        fields = ['name', 'level']

class StudentProfileSerializer(serializers.ModelSerializer):
    skills = UserSkillSerializer(many=True, source='userskill_set')
    
    class Meta:
        model = StudentProfile
        fields = ['id', 'name', 'contact', 'domain', 'role_preference', 
                 'profile_picture', 'projects', 'linkedin', 'github',
                 'portfolio', 'certifications', 'is_public', 'experience',
                 'bio', 'skills']