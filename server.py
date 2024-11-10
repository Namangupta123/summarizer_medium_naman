from flask import Flask, request, jsonify
import json
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv
import traceback
from functools import wraps
import jwt
from google.oauth2 import id_token
from google.auth.transport import requests
import requests as http_requests 
from werkzeug.serving import WSGIRequestHandler
load_dotenv()

WSGIRequestHandler.protocol_version = "HTTP/1.1"
app = Flask(__name__)
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"],
     max_age=3600)

mistral_key = os.getenv("mistral_key")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]=os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGSMITH_API")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGSMITH_PROJECT")
template = """
You are an AI summarization agent tasked with summarizing the provided content. Your goal is to create a concise and informative summary that highlights the key points and important notes. Please follow these instructions:

Instructions
1. Content Analysis: Carefully read and analyze the provided content.
2. Summary Creation: Write a summary that captures the main ideas and essential details. Ensure the summary is clear and concise.
3. Important Notes: Identify and list any critical notes or insights that should be emphasized.
4. Bullet Points: Present the summary and notes using bullet points for clarity and easy reading.

Input Variables
- {content}: The text or document you want summarized.

Example:
Input
{content}: "The rapid advancement of artificial intelligence has sparked both excitement and concern across various industries. While AI promises increased efficiency and innovative solutions, it also raises questions about job displacement and ethical implications."

Output
- Main Ideas: AI advancements bring both opportunities and challenges.
- Efficiency and Innovation: AI offers increased efficiency and innovative solutions.
- Concerns: Raises questions about job displacement and ethical implications.

Please ensure that the summary is accurate and reflects the original content's intent.
"""

llm = ChatMistralAI(model="mistral-large-latest", temperature=0.5, mistral_api_key=mistral_key, max_tokens=2000)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI summarization expert. Your primary function is to distill complex information into clear, concise summaries while maintaining the original intent and key details. Use a neutral and informative tone."),
    ("human", template),
])

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
            print(json.dumps(token))
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

@app.route('/summarize', methods=['POST', 'OPTIONS'])
@verify_token
def summarize():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
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
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True, timeout=180)