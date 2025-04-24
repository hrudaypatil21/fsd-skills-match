# views.py
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from .models import StudentProfile, UserSkill, CachedEmbedding
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from django.core.cache import cache
from pymongo import MongoClient

mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://vashninadar123:mibrRJ65Zk3gqFIc@cluster0.yxkud.mongodb.net/test?retryWrites=true&w=majority&appName=Cluster0")
client = MongoClient(mongo_uri)
db = client.get_database()
student_profiles = db['studentprofiles']
mentors = db['mentors']

model = SentenceTransformer('all-mpnet-base-v2')

User = get_user_model()


skill_role_map = {
    'python': ['data', 'backend', 'automation', 'machine learning'],
    'javascript': ['web', 'frontend', 'fullstack', 'mobile'],
    'java': ['backend', 'mobile', 'enterprise'],
    'c#': ['backend', 'game development', 'enterprise'],
    'go': ['backend', 'cloud', 'devops'],
    'rust': ['systems', 'backend', 'embedded'],
    'kotlin': ['mobile', 'backend'],
    'swift': ['mobile', 'ios'],
    'react': ['web', 'frontend', 'mobile'],
    'angular': ['web', 'frontend', 'enterprise'],
    'vue': ['web', 'frontend'],
    'node.js': ['backend', 'fullstack', 'web'],
    'express': ['backend', 'web'],
    'django': ['backend', 'web', 'fullstack'],
    'flask': ['backend', 'web', 'microservices'],
    'spring': ['backend', 'enterprise'],
    'react native': ['mobile', 'cross-platform'],
    'flutter': ['mobile', 'cross-platform'],
    'android': ['mobile', 'android'],
    'ios': ['mobile', 'ios'],
    'sql': ['data', 'backend', 'analytics'],
    'nosql': ['data', 'backend', 'big data'],
    'pandas': ['data', 'analytics', 'machine learning'],
    'numpy': ['data', 'machine learning', 'scientific computing'],
    'tensorflow': ['machine learning', 'deep learning'],
    'pytorch': ['machine learning', 'deep learning'],
    'spark': ['big data', 'data engineering'],
    'hadoop': ['big data', 'data engineering'],
    'docker': ['devops', 'cloud', 'backend'],
    'kubernetes': ['devops', 'cloud', 'scalability'],
    'aws': ['cloud', 'devops', 'backend'],
    'azure': ['cloud', 'devops', 'enterprise'],
    'gcp': ['cloud', 'devops', 'machine learning'],
    'terraform': ['devops', 'cloud', 'infrastructure'],
    'ansible': ['devops', 'automation'],
    'jenkins': ['devops', 'ci/cd'],
    'graphql': ['backend', 'api', 'web'],
    'rest': ['backend', 'api', 'web'],
    'linux': ['systems', 'devops', 'backend'],
    'bash': ['systems', 'devops', 'automation'],
    'git': ['version control', 'collaboration'],
    'selenium': ['testing', 'automation'],
    'cypress': ['testing', 'frontend'],
    'figma': ['design', 'ux/ui', 'frontend'],
    'sketch': ['design', 'ux/ui'],
    'adobe xd': ['design', 'ux/ui'],
    'unity': ['game development', 'ar/vr'],
    'unreal engine': ['game development', 'ar/vr'],
}

def get_cached_embedding(text):
    cache_key = f"embedding_{text}"
    cached = cache.get(cache_key)
    if cached is not None:
        return np.frombuffer(cached, dtype='float32')
    
    try:
        db_cached = CachedEmbedding.objects.get(text=text)
        cache.set(cache_key, db_cached.embedding, timeout=60*60*24)
        return np.frombuffer(db_cached.embedding, dtype='float32')
    except CachedEmbedding.DoesNotExist:
        embedding = model.encode(text)
        embedding_bytes = embedding.tobytes()
        
        CachedEmbedding.objects.create(
            text=text,
            embedding=embedding_bytes
        )
        
        cache.set(cache_key, embedding_bytes, timeout=60*60*24)
        return embedding

def get_complementary_skills(target_skill, top_n=10, cluster_weight=0.7):
    target_clusters = set(skill_role_map.get(target_skill, []))
    
    candidates = []
    for skill, clusters in skill_role_map.items():
        if skill == target_skill:
            continue

        common_clusters = target_clusters.intersection(clusters)
        cluster_score = 1 - len(common_clusters)/max(1, len(target_clusters))

        candidates.append({
            'skill': skill,
            'clusters': clusters,
            'cluster_score': cluster_score
        })

    candidate_skills = [c['skill'] for c in candidates]
    candidate_embeddings = model.encode(candidate_skills)
    target_embedding = model.encode(target_skill)

    similarities = cosine_similarity([target_embedding], candidate_embeddings)[0]
    for i, c in enumerate(candidates):
        c['similarity'] = similarities[i]

    for c in candidates:
        c['composite_score'] = (
            (1 - cluster_weight) * c['similarity'] +
            cluster_weight * c['cluster_score']
        )

    results = sorted(candidates, key=lambda x: -x['composite_score'])
    return [r['skill'] for r in results[:top_n]]

@api_view(['GET'])
def get_student_profile(request, uid):
    try:
        profile = student_profiles.find_one({"uid": uid})
        if not profile:
            return Response({"error": "Profile not found"}, status=404)
        
        # Handle skills extraction robustly
        skills = []
        raw_skills = profile.get("skills", [])
        
        if isinstance(raw_skills, list):
            for item in raw_skills:
                if isinstance(item, dict) and 'name' in item:
                    skills.append(item['name'].lower())
                elif isinstance(item, str):
                    skills.append(item.lower())
        
        # Transform the profile data
        transformed = {
            "uid": profile.get("uid"),
            "name": profile.get("name"),
            "skills": skills,
            "profilePicture": profile.get("profilePicture", ""),
            "rolePreference": profile.get("rolePreference", ""),
            "domain": profile.get("domain", "")
        }
        
        return Response(transformed)
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
@api_view(['POST'])
def find_mentors(request, uid):
    try:
        user_skills = request.data.get('skills', [])
        current_user_id = request.data.get('userId')
        
        if not user_skills:
            return Response({"error": "No skills provided"}, status=400)
        
        # Get all mentors
        profile = mentors.find_one({"uid": uid})
        if not profile:
            return Response({"error": "Profile not found"}, status=404)
        
        # Prepare response data
        response_data = {
            'mentors': [],
            'user_skills': user_skills
        }
        
        # Calculate skill matches
        user_skills_text = ' '.join(user_skills)
        user_embedding = model.encode([user_skills_text])[0]
        
        for mentor in mentors:
            mentor_skills = mentor.get('skills', [])
            if mentor_skills:
                mentor_skills_text = ' '.join(mentor_skills)
                mentor_embedding = model.encode([mentor_skills_text])[0]
                similarity = float(cosine_similarity(
                    [user_embedding],
                    [mentor_embedding]
                )[0][0])
                
                # Calculate skill overlap
                skill_overlap = set(s.lower() for s in user_skills).intersection(
                    set(s.lower() for s in mentor_skills)
                )
                
                response_data['mentors'].append({
                    'mentor_id': str(mentor['_id']),
                    'name': mentor.get('name', ''),
                    'skills': mentor_skills,
                    'profile_picture': mentor.get('profilePicture', ''),
                    'domain': mentor.get('domain', ''),
                    'current_position': mentor.get('currentPosition', ''),
                    'similarity_score': similarity,
                    'skill_overlap': list(skill_overlap),
                    'overlap_count': len(skill_overlap)
                })
        
        # Sort by similarity and overlap
        response_data['mentors'] = sorted(
            response_data['mentors'],
            key=lambda x: (-x['similarity_score'], -x['overlap_count'])
        )[:20]
        
        return Response(response_data)
    
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['POST'])
def find_complementary_teammates(request):
    try:
        print("\n=== Received recommendation request ===")
        request_data = request.data
        print("Request data:", request_data)
        
        # Get data from request
        user_skills = request_data.get('skills', [])
        current_user_id = request_data.get('userId')
        
        print(f"User ID: {current_user_id}, Skills: {user_skills}")

        if not isinstance(user_skills, list):
            print("Error: Skills is not a list")
            return Response({"error": "Skills should be an array"}, status=400)
            
        if not user_skills:
            print("Warning: Empty skills array")
            return Response({
                'by_skill': {},
                'similar_users': [],
                'user_skills': []
            })

        user_skills = request.data.get('skills', [])
        current_user_id = request.data.get('userId')
        
        if not user_skills:
            return Response({"error": "No skills provided"}, status=400)
        
        

        all_users = list(student_profiles.find({"_id": {"$ne": current_user_id}}))

        response_data = {
            'by_skill': defaultdict(list),
            'similar_users': [],
            'user_skills': user_skills
        }

        for skill in user_skills:
            complementary_skills = get_complementary_skills(skill)
            
            for other_user in all_users:
                other_skills = [s['name'].lower() if isinstance(s, dict) else s.lower() 
                              for s in other_user.get('skills', [])]
                common_skills = set(other_skills).intersection(complementary_skills)
                
                if common_skills:
                    response_data['by_skill'][skill].append({
                        'user_id': str(other_user['_id']),
                        'name': other_user.get('name', ''),
                        'matching_skills': list(other_skills),
                        'profile_picture': other_user.get('profilePicture', ''),
                        'role_preference': other_user.get('rolePreference', ''),
                        'domain': other_user.get('domain', '')
                    })

        user_skills_text = ' '.join(user_skills)
        user_embedding = model.encode([user_skills_text])[0]
        
        other_users_data = []
        for other_user in all_users:
            other_skills = [s['name'].lower() if isinstance(s, dict) else s.lower() 
                          for s in other_user.get('skills', [])]
            if other_skills:
                other_skills_text = ' '.join(other_skills)
                other_embedding = model.encode([other_skills_text])[0]
                similarity = float(cosine_similarity(
                    [user_embedding],
                    [other_embedding]
                )[0][0])
                
                other_users_data.append({
                    'user_id': str(other_user['_id']),
                    'name': other_user.get('name', ''),
                    'skills': other_skills,
                    'profile_picture': other_user.get('profilePicture', ''),
                    'role_preference': other_user.get('rolePreference', ''),
                    'domain': other_user.get('domain', ''),
                    'similarity_score': similarity
                })
        
        # Sort by similarity and limit to top 20
        response_data['similar_users'] = sorted(
            other_users_data,
            key=lambda x: -x['similarity_score']
        )[:20]
        
        return Response(response_data)
    
    except Exception as e:
        return Response({"error": str(e)}, status=500)

