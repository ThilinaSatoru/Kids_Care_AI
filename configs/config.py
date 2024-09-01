# config.py
import os
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import db

# ===================================================================================================
date_now = datetime.now()

VIDEO_HEIGHT = 640
VIDEO_WIDTH = 480

# ===================================================================================================
# FALL - YOLO Model Path
YOLO_MODEL_PATH = '/fall-ml/falldetectionmodelv3.pt'

# Create directories for saving images
IMG_OUTPUT_DIRECTORY = 'detected'
os.makedirs(IMG_OUTPUT_DIRECTORY, exist_ok=True)

# Firebase Configuration
GOOGLE_APPLICATION_CREDENTIALS = 'kidzcare-97f3c-firebase-adminsdk-fpml6-3904295b4d.json'
FIREBASE_STORAGE_BUCKET = 'kidzcare-97f3c.appspot.com'
FIREBASE_DATABASE_NAME = '/kiddycare'
FIREBASE_DATABASE_URL = 'https://kidzcare-97f3c-default-rtdb.firebaseio.com'

# Initialize Firebase
cred = credentials.Certificate(GOOGLE_APPLICATION_CREDENTIALS)
firebase_admin.initialize_app(cred, {
    'storageBucket': FIREBASE_STORAGE_BUCKET,
    'databaseURL': FIREBASE_DATABASE_URL
})

firebase_ref = db.reference(FIREBASE_DATABASE_NAME)
# Initialize Firebase Storage bucket
firebase_bucket = storage.bucket()
# ===================================================================================================
