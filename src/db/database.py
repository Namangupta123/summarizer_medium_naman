from sqlalchemy import create_engine, text, exc
from datetime import datetime
import os
import pytz

# Configure timezone
IST = pytz.timezone('Asia/Kolkata')

# Database setup
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are not set")

# Format the connection URL for SQLAlchemy
# Example Supabase URL format: postgresql://postgres:[PASSWORD]@db.[PROJECT_ID].supabase.co:5432/postgres
DATABASE_URL = f"{SUPABASE_URL}?apikey={SUPABASE_KEY}"
engine = create_engine(DATABASE_URL)

def init_db():
    try:
        with engine.connect() as conn:
            # Drop the existing table if it exists
            conn.execute(text("DROP TABLE IF EXISTS users"))
            conn.commit()
            
            # Create the table with the new schema
            conn.execute(text("""
                CREATE TABLE users (
                    email VARCHAR(255) PRIMARY KEY,
                    summary_count INTEGER DEFAULT 5,
                    last_reset DATE DEFAULT CURRENT_DATE,
                    welcome_email_sent BOOLEAN DEFAULT FALSE
                )
            """))
            conn.commit()
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        raise

def get_remaining_summaries(email, conn):
    """Calculate remaining summaries for a user"""
    try:
        result = conn.execute(
            text("SELECT * FROM users WHERE email = :email"),
            {"email": email}
        ).fetchone()
        
        if not result:
            return None
            
        current_date = datetime.now(IST).date()
        last_reset_date = result.last_reset
        
        # Check if reset is needed (different date)
        needs_reset = current_date > last_reset_date
        remaining = result.summary_count if not needs_reset else 5
        
        return {
            'remaining': remaining,
            'needs_reset': needs_reset,
            'is_new_user': False,
            'last_reset': last_reset_date
        }
    except Exception as e:
        print(f"Error in get_remaining_summaries: {str(e)}")
        return None

def check_summary_limit(email):
    try:
        with engine.connect() as conn:
            # Check if user exists
            summary_info = get_remaining_summaries(email, conn)
            
            if summary_info is None:
                # New user registration - they get 5 summaries
                conn.execute(
                    text("""
                        INSERT INTO users (email, summary_count, last_reset, welcome_email_sent) 
                        VALUES (:email, 5, CURRENT_DATE, TRUE)
                    """),
                    {"email": email}
                )
                conn.commit()
                return True
            
            # Reset counter if needed
            if summary_info['needs_reset']:
                conn.execute(
                    text("""
                        UPDATE users 
                        SET summary_count = 5, 
                            last_reset = CURRENT_DATE
                        WHERE email = :email
                    """),
                    {"email": email}
                )
                conn.commit()
                return True
            
            # Check if they have summaries remaining
            return summary_info['remaining'] > 0
            
    except Exception as e:
        print(f"Error checking summary limit: {str(e)}")
        return False

def decrement_summary_count(email):
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
        print(f"Error decrementing summary count: {str(e)}")

def get_user_summary_count(email, conn):
    """Get user's summary count and handle date reset"""
    try:
        user = conn.execute(
            text("SELECT * FROM users WHERE email = :email"),
            {"email": email}
        ).fetchone()
        
        if not user:
            return None
            
        result = conn.execute(
            text("""
                SELECT 
                    CASE 
                        WHEN CURRENT_DATE > last_reset
                        THEN 5
                        ELSE summary_count
                    END as current_count,
                    last_reset
                FROM users 
                WHERE email = :email
            """),
            {"email": email}
        ).fetchone()
        
        return result
    except Exception as e:
        print(f"Error in get_user_summary_count: {str(e)}")
        return None