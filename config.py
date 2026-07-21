import os
import mysql.connector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

def load_env():
    """
    Zero-dependency .env file parser to load variables into os.environ.
    Ensures seamless offline execution without needing extra package installations.
    """
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        k = parts[0].strip()
                        v = parts[1].strip().strip('"').strip("'")
                        os.environ[k] = v

# Load settings
load_env()

DB_HOST = os.environ.get("DB_HOST", "b7qvuwrkes3idbbzs52o-mysql.services.clever-cloud.com")
DB_USER = os.environ.get("DB_USER", "uhxyp3ooh8dyea1m")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "u1T1ypwg5Fe8rrKTgVsy")
DB_NAME = os.environ.get("DB_NAME", "b7qvuwrkes3idbbzs52o")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))

def get_db_connection():
    """
    Returns a MySQL connection using parameters configured in the .env file.
    """
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
