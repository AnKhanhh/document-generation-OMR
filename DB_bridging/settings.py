from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SECRET_KEY = 'omr-database-bridge-key'
DEBUG = True
USE_TZ = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'app.sqlite',
    }
}

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'DB_bridging',
]

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
