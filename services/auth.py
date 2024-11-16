from functools import wraps
from flask import request, jsonify
import requests
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

def verify_google_token(token):
    try:
        # First try with userinfo endpoint
        userinfo_url = 'https://www.googleapis.com/oauth2/v2/userinfo'
        headers = {'Authorization': f'Bearer {token}'}
        
        response = requests.get(userinfo_url, headers=headers)
        
        if response.ok:
            user_info = response.json()
            if 'email' in user_info:
                return user_info
                
        # If userinfo fails, try token verification
        token_url = f'https://oauth2.googleapis.com/tokeninfo?access_token={token}'
        token_response = requests.get(token_url)
        
        if token_response.ok:
            token_info = token_response.json()
            if 'email' in token_info and 'error' not in token_info:
                return token_info
        
        return None
    except Exception as e:
        print(f"Token verification error: {str(e)}")
        return None

def verify_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({'message': 'Authorization header is missing'}), 401
            
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({'message': 'Invalid Authorization header format'}), 401
            
        token = parts[1]
        user_info = verify_google_token(token)
        
        if not user_info:
            return jsonify({'message': 'Invalid or expired token'}), 401
            
        if 'email' not in user_info:
            return jsonify({'message': 'Email not found in token'}), 401
            
        request.user = user_info
        return f(*args, **kwargs)
    
    return decorated