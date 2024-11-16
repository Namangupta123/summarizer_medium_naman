
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import pytz
import os
from services.email import send_welcome_email

# Configure timezone
IST = pytz.timezone('Asia/Kolkata')

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
                
                return {
                    "count": 0,
                    "limit": 5,
                    "remaining": 5,
                    "can_summarize": True
                }
            
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
                return {
                    "count": 0,
                    "limit": 5,
                    "remaining": 5,
                    "can_summarize": True
                }
            
            return {
                "count": 5 - result.summary_count,
                "limit": 5,
                "remaining": result.summary_count,
                "can_summarize": result.summary_count > 0,
                "last_reset": last_reset.isoformat()
            }
            
    except Exception as e:
        print(f"Error checking summary limit: {str(e)}")
        return {
            "count": 0,
            "limit": 5,
            "remaining": 0,
            "can_summarize": False,
            "error": str(e)
        }

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
            return True
    except Exception as e:
        print(f"Error incrementing summary count: {str(e)}")
        return False