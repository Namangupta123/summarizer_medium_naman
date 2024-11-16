from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.serving import WSGIRequestHandler
from dotenv import load_dotenv
from services.auth import verify_token
from services.cache import init_redis, get_cached_summary, cache_summary
from services.database import init_db, check_summary_limit, increment_summary_count
from services.summarizer import get_summary
import os

load_dotenv()

WSGIRequestHandler.protocol_version = "HTTP/1.1"
app = Flask(__name__)
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type", "Authorization", "Accept"],
     methods=["GET", "POST", "OPTIONS"],
     max_age=3600)

# Initialize services
init_db()
init_redis()

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
            
        count_info = check_summary_limit(email)
        return jsonify(count_info)
            
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
        if not check_summary_limit(email)['can_summarize']:
            return jsonify({
                "error": "Daily summary limit reached (5/5). Please try again tomorrow.",
                "limit_reached": True
            }), 429

        content = request.json.get('content')
        if not content:
            return jsonify({"error": "No content provided"}), 400

        # Check cache first
        cached_summary = get_cached_summary(content)
        if cached_summary:
            increment_summary_count(email)
            return jsonify({"summary": cached_summary, "cached": True})

        # Generate new summary
        summary = get_summary(content)
        if summary:
            cache_summary(content, summary)
            increment_summary_count(email)
            return jsonify({"summary": summary, "cached": False})
        
        return jsonify({"error": "Failed to generate summary"}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True, timeout=180)