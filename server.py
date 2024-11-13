from flask import Flask, request, jsonify
import json
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from functools import wraps
import jwt
from google.oauth2 import id_token
from google.auth.transport import requests
import requests as http_requests 
from werkzeug.serving import WSGIRequestHandler
from langchain_openai import AzureChatOpenAI
from sqlalchemy import create_engine, text
from datetime import datetime
load_dotenv()

WSGIRequestHandler.protocol_version = "HTTP/1.1"
app = Flask(__name__)
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"],
     max_age=3600)

openai_endpoint=os.getenv("OPENAI_ENDPOINT")
openai_api=os.getenv("OPENAI_API")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]=os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGSMITH_API")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGSMITH_PROJECT")
os.environ["OPENAI_API_VERSION"] = "2024-08-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API")

# Database setup with Vercel Postgres
POSTGRES_URL = os.getenv('POSTGRES_URL')
if not POSTGRES_URL:
    raise ValueError("POSTGRES_URL environment variable is not set")

# Modify the URL to use postgresql instead of postgres
POSTGRES_URL = POSTGRES_URL.replace('postgres://', 'postgresql://')
engine = create_engine(POSTGRES_URL)

# Create users table if it doesn't exist
def init_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    email VARCHAR(255) PRIMARY KEY,
                    summary_count INTEGER DEFAULT 0,
                    last_reset TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        raise

# Initialize database on startup
init_db()

template = """
You are an expert summarization AI. Your role is to distill complex information into clear, concise summaries that capture the essence of the content.

### Instructions
1. **Content Analysis**: Carefully read and analyze the provided content.
2. **Summary Creation**: Write a summary that captures the main ideas and essential details. Ensure the summary is clear and concise.
3. **Important Notes**: Identify and list any critical notes or insights that should be emphasized.
4. **Bullet Points**: Present the summary and notes using bullet points for clarity and easy reading.

### Input
- {content}: The text or document you want summarized.

### Examples
- **Example 1**: Given a text about climate change, summarize the main causes and effects in bullet points.
- **Example 2**: For a document on the history of the internet, highlight key milestones and technological advancements.

Please ensure that the summary is accurate and reflects the original content's intent.
"""

llm = AzureChatOpenAI(
    openai_api_version="2024-08-01-preview",
    azure_endpoint=openai_endpoint,
    openai_api_key=openai_api,
    deployment_name="gpt-4",
    temperature=0.3,
    max_retries=1,
    max_tokens=900,
    model_version="turbo-2024-04-09",
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI summarization expert. Your primary function is to distill complex information into clear, concise summaries while maintaining the original intent and key details. Use a neutral and informative tone."),
    ("human", template),
])

def check_summary_limit(email):
    try:
        with engine.connect() as conn:
            # Get user record or create if doesn't exist
            result = conn.execute(
                text("SELECT * FROM users WHERE email = :email"),
                {"email": email}
            ).fetchone()
            
            if not result:
                conn.execute(
                    text("INSERT INTO users (email, summary_count) VALUES (:email, 0)"),
                    {"email": email}
                )
                conn.commit()
                return True
            
            # Check if user has reached limit
            if result.summary_count >= 5:
                # Check if it's time to reset (e.g., after 24 hours)
                last_reset = result.last_reset
                if (datetime.now() - last_reset).days >= 1:
                    conn.execute(
                        text("UPDATE users SET summary_count = 0, last_reset = CURRENT_TIMESTAMP WHERE email = :email"),
                        {"email": email}
                    )
                    conn.commit()
                    return True
                return False
            
            return True
    except Exception as e:
        print(f"Error checking summary limit: {str(e)}")
        return False

def increment_summary_count(email):
    try:
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE users SET summary_count = summary_count + 1 WHERE email = :email"),
                {"email": email}
            )
            conn.commit()
    except Exception as e:
        print(f"Error incrementing summary count: {str(e)}")

def verify_google_token(token):
    try:
        response = http_requests.get(
            'https://oauth2.googleapis.com/tokeninfo',
            params={'access_token': token}
        )
        if response.status_code != 200:
            return None
        
        token_info = response.json()
        if token_info.get('error'):
            return None
            
        return token_info
    except Exception as e:
        print(f"Token verification error: {str(e)}")
        return None

def verify_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            token = token.split('Bearer ')[1]
            user_info = verify_google_token(token)
            if not user_info:
                return jsonify({'message': 'Invalid token'}), 401
            request.user = user_info
        except:
            return jsonify({'message': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def home():
    return jsonify({"status": "alive", "message": "Medium Summarizer API is running"})

@app.route('/user/summary-count', methods=['GET'])
@verify_token
def get_summary_count():
    try:
        email = request.user.get('email')
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT summary_count FROM users WHERE email = :email"),
                {"email": email}
            ).fetchone()
            count = result.summary_count if result else 0
            return jsonify({"count": count, "limit": 5})
    except Exception as e:
        print(f"Error getting summary count: {str(e)}")
        return jsonify({"error": "Failed to get summary count"}), 500

@app.route('/summarize', methods=['POST', 'OPTIONS'])
@verify_token
def summarize():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        email = request.user.get('email')
        if not check_summary_limit(email):
            return jsonify({
                "error": "Daily summary limit reached (5/5). Please try again tomorrow.",
                "limit_reached": True
            }), 429

        content = request.json['content']
        input_data = {
            "content": content
        }
        response = (
            prompt
            | llm.bind(stop=["\nsummarization"])
            | StrOutputParser()
        )
        summary = response.invoke(input_data)
        
        # Increment the summary count after successful generation
        increment_summary_count(email)
        
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True, timeout=180)