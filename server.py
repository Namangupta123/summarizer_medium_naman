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
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

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

# SendGrid configuration
SENDGRID_API_KEY = os.getenv('SEND_GRID_AP')
FROM_EMAIL = os.getenv('FROM_EMAIL')

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
                    last_reset TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    welcome_email_sent BOOLEAN DEFAULT FALSE
                )
            """))
            conn.commit()
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        raise

# Initialize database on startup
init_db()

def send_welcome_email(email):
    try:
        message = Mail(
            from_email=FROM_EMAIL,
            to_emails=email,
            subject='Welcome to Medium Blog Summarizer!',
            html_content=f'''
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h2>Welcome to Medium Blog Summarizer! 🎉</h2>
                    <p>Dear User,</p>
                    <p>Thank you for choosing Medium Blog Summarizer! We're excited to have you on board.</p>
                    <p>With our tool, you can:</p>
                    <ul>
                        <li>Get AI-powered summaries of Medium articles</li>
                        <li>Save time while staying informed</li>
                        <li>Access 5 free summaries daily</li>
                    </ul>
                    <p>Start summarizing your first article today!</p>
                    <p>Best regards,<br>The Medium Blog Summarizer Team</p>
                </div>
            '''
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
        return True
    except Exception as e:
        print(f"Error sending welcome email: {str(e)}")
        return False

template = """
You are an expert summarization AI. Your role is to create well-structured HTML summaries that are clean and semantic. 

## Instructions
1. Create a summary with the following sections:
   - Main Summary (brief overview)
   - Key Points
   - Important Details
   - Takeaways

2. Use semantic HTML elements for structure. The response should follow this format:

<article>
    <section>
        <h2>Main Summary</h2>
        <p>[Concise overview here]</p>
    </section>

    <section>
        <h2>Key Points</h2>
        <ul>
            <li>[Key point 1]</li>
            <li>[Key point 2]</li>
        </ul>
    </section>

    <section>
        <h2>Important Details</h2>
        <h3>Context</h3>
        <ul>
            <li>[Detail 1]</li>
        </ul>
    </section>

    <section>
        <h2>Key Takeaways</h2>
        <ul>
            <li>[Takeaway 1]</li>
        </ul>
    </section>
</article>

## Input Content to Summarize:
{content}

Ensure the summary is comprehensive yet concise, with proper semantic HTML structure throughout. 

## Examples
- **Example 1**: If summarizing a news article, the Main Summary should provide a brief overview of the event, Key Points should list the main facts, Important Details should provide context such as background information, and Key Takeaways should highlight the implications or future outlook.

- **Example 2**: For a research paper, the Main Summary should encapsulate the research question and findings, Key Points should outline the methodology and results, Important Details should delve into the data analysis, and Key Takeaways should discuss the significance of the findings.
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
    ("system", "You are an AI specialized in creating structured HTML summaries. Your task is to transform complex information into clear, concise, and well-organized summaries using semantic HTML. Maintain a neutral and informative tone, ensuring that each section is distinct and logically ordered. Prioritize clarity and coherence in your summaries."),
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
                # New user registration
                conn.execute(
                    text("""
                        INSERT INTO users (email, summary_count, last_reset, welcome_email_sent) 
                        VALUES (:email, 5, CURRENT_TIMESTAMP AT TIME ZONE 'Asia/Kolkata', FALSE)
                    """),
                    {"email": email}
                )
                conn.commit()
                
                # Send welcome email
                if send_welcome_email(email):
                    conn.execute(
                        text("UPDATE users SET welcome_email_sent = TRUE WHERE email = :email"),
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