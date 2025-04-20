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

mongo_uri = os.getenv("MONGO_URI", "#mongodb link#")
client = MongoClient(mongo_uri)
db = client.get_database()
student_profiles = db['studentprofiles']

model = SentenceTransformer('all-mpnet-base-v2')

User = get_user_model()
model = SentenceTransformer('all-mpnet-base-v2')


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

def get_complementary_skills(target_skill, top_n=10, cluster_weight=0.5):
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

@api_view(['POST'])
def find_complementary_teammates(request):
    try:
        user_skills = request.data.get('skills', [])
        current_user_id = request.data.get('userId')
        
        if not user_skills:
            return Response({"error": "No skills provided"}, status=400)
        
        # Get all users except current user
        all_users = list(student_profiles.find({"_id": {"$ne": current_user_id}}))
        
        # Prepare response data
        response_data = {
            'by_skill': defaultdict(list),
            'similar_users': [],
            'user_skills': user_skills
        }
        
        # Calculate complementary skills matches
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
                        'matching_skills': list(common_skills),
                        'profile_picture': other_user.get('profilePicture', ''),
                        'role_preference': other_user.get('rolePreference', ''),
                        'domain': other_user.get('domain', '')
                    })
        
        # Calculate overall similarity
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
