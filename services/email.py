from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import os

FROM_EMAIL = os.getenv('FROM_EMAIL')
SENDGRID_API_KEY = os.getenv('SEND_GRID_API')

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