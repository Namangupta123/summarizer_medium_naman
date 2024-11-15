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
from datetime import datetime, timedelta
import pytz

load_dotenv()

WSGIRequestHandler.protocol_version = "HTTP/1.1"
app = Flask(__name__)
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization", "Accept"],
     methods=["GET", "POST", "OPTIONS"],
     max_age=3600)

# Configure timezone
IST = pytz.timezone('Asia/Kolkata')

openai_endpoint = os.getenv("OPENAI_ENDPOINT")
openai_api = os.getenv("OPENAI_API")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["OPENAI_API_VERSION"] = "2024-08-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("OPENAI_API")

# Database setup
POSTGRES_URL = os.getenv('POSTGRES_URL')
if not POSTGRES_URL:
    raise ValueError("POSTGRES_URL environment variable is not set")

POSTGRES_URL = POSTGRES_URL.replace('postgres://', 'postgresql://')
engine = create_engine(POSTGRES_URL)

def init_db():
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    email VARCHAR(255) PRIMARY KEY,
                    summary_count INTEGER DEFAULT 5,
                    last_reset TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        raise

# Initialize database on startup
init_db()

template = """
You are an expert summarization AI. Your role is to create beautifully formatted, HTML summaries with clean CSS styling.

### System Prompt
You are a summarization AI specialized in generating HTML content with embedded CSS. Your task is to transform input text into a structured summary with specific sections, ensuring a consistent and visually appealing format.

### Instructions
1. Create a summary with the following sections:
   - Main Summary (brief overview)
   - Key Points
   - Important Details
   - Takeaways

2. Use HTML with embedded CSS styling. The response should start with:
```html
<style>
.summary-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    line-height: 1.6;
    color: #2c3e50;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
.summary-section {
    margin-bottom: 24px;
}
h2 {
    color: #1a365d;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
    margin-top: 24px;
}
h3 {
    color: #2d3748;
    margin-top: 16px;
}
p {
    margin: 16px 0;
}
ul {
    padding-left: 24px;
    margin: 16px 0;
}
li {
    margin: 8px 0;
    line-height: 1.5;
}
.highlight {
    background-color: #faf5ff;
    padding: 16px;
    border-radius: 8px;
    border-left: 4px solid #805ad5;
}
.key-point {
    font-weight: 500;
    color: #2c5282;
}
</style>
```

### Input
```markdown
{content}
```

### Response Format Example
```html
<div class="summary-container">
    <div class="summary-section">
        <h2>Main Summary</h2>
        <div class="highlight">
            <p>[Concise overview here]</p>
        </div>
    </div>

    <div class="summary-section">
        <h2>Key Points</h2>
        <ul>
            <li class="key-point">[Key point 1]</li>
            <li class="key-point">[Key point 2]</li>
        </ul>
    </div>

    <div class="summary-section">
        <h2>Important Details</h2>
        <div>
            <h3>Context</h3>
            <ul>
                <li>[Detail 1]</li>
            </ul>
        </div>
    </div>

    <div class="summary-section">
        <h2>Key Takeaways</h2>
        <ul>
            <li>[Takeaway 1]</li>
        </ul>
    </div>
</div>
```

### Input Variables
- {content}: The text content that needs to be summarized.
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

def verify_google_token(token):
    try:
        # First try with userinfo endpoint
        userinfo_url = 'https://www.googleapis.com/oauth2/v2/userinfo'
        headers = {'Authorization': f'Bearer {token}'}
        
        response = http_requests.get(userinfo_url, headers=headers)
        
        if response.ok:
            user_info = response.json()
            if 'email' in user_info:
                return user_info
                
        # If userinfo fails, try token verification
        token_url = f'https://oauth2.googleapis.com/tokeninfo?access_token={token}'
        token_response = http_requests.get(token_url)
        
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

def check_summary_limit(email):
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM users WHERE email = :email"),
                {"email": email}
            ).fetchone()
            
            if not result:
                conn.execute(
                    text("""
                        INSERT INTO users (email, summary_count, last_reset) 
                        VALUES (:email, 5, CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Kolkata')
                    """),
                    {"email": email}
                )
                conn.commit()
                return True
            
            last_reset = result.last_reset.astimezone(IST)
            current_time = datetime.now(IST)
            
            if current_time - last_reset >= timedelta(days=1):
                conn.execute(
                    text("""
                        UPDATE users 
                        SET summary_count = 5, 
                            last_reset = CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Kolkata'
                        WHERE email = :email
                    """),
                    {"email": email}
                )
                conn.commit()
                return True
            
            return result.summary_count > 0
            
    except Exception as e:
        print(f"Error checking summary limit: {str(e)}")
        return False

def increment_summary_count(email):
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE users 
                    SET summary_count = summary_count - 1 
                    WHERE email = :email
                """),
                {"email": email}
            )
            conn.commit()
    except Exception as e:
        print(f"Error incrementing summary count: {str(e)}")

@app.route('/')
def home():
    return jsonify({"status": "alive", "message": "Medium Summarizer API is running"})

@app.route('/user/summary-count', methods=['GET'])
@verify_token
def get_summary_count():
    try:
        email = request.user.get('email')
        if not email:
            return jsonify({"error": "Email not found in token"}), 400
            
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT summary_count, last_reset,
                    CASE 
                        WHEN (CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Kolkata' - last_reset) >= INTERVAL '1 day'
                        THEN 5
                        ELSE summary_count
                    END as current_count
                    FROM users 
                    WHERE email = :email
                """),
                {"email": email}
            ).fetchone()
            
            if not result:
                return jsonify({"count": 0, "limit": 5, "remaining": 5})
            
            current_count = result.current_count
            
            # Convert last_reset to IST before sending
            last_reset_ist = result.last_reset.astimezone(IST)
            
            return jsonify({
                "count": 5 - current_count,
                "limit": 5,
                "remaining": current_count,
                "last_reset": last_reset_ist.isoformat()
            })
            
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

        content = request.json.get('content')
        if not content:
            return jsonify({"error": "No content provided"}), 400

        input_data = {"content": content}
        response = (
            prompt
            | llm.bind(stop=["\nsummarization"])
            | StrOutputParser()
        )
        summary = response.invoke(input_data)
        
        increment_summary_count(email)
        
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True, timeout=180)