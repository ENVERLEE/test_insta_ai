import os

class Config:
    """Base configuration class."""

    # Secret key for CSRF protection
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'your_default_secret_key')  # Set a strong secret key

    # SQLAlchemy settings (if you're using SQLAlchemy)
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///site.db')  # Default to SQLite if not set
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # Disable modification tracking to save resources

    # Other configuration settings
    DEBUG = os.getenv('FLASK_DEBUG', 'False') == 'True'  # Debug mode from environment variable
    TESTING = os.getenv('FLASK_TESTING', 'False') == 'True'  # Testing mode from environment variable

    # Hugging Face API settings (example, adjust as necessary)
    HF_TOKEN = os.getenv('HF_TOKEN')  # Ensure this is set in your environment variables

    # Path for file uploads or other file-related settings (if needed)
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB file size limit

    # Add other settings here as needed

    # Optional: Logging configuration
    LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO')  # Set the logging level
    LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
