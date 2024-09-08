import logging

from config import *

logging.basicConfig(level=logging.DEBUG)

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate('kidzcare-97f3c-9fd483c9d2c4.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'kidzcare-97f3c.appspot.com',
        'databaseURL': 'https://kidzcare-97f3c-default-rtdb.firebaseio.com'
    })

try:
    print("Starting...")

    # Push data to Firebase Realtime Database
    users_ref = firebase_ref.child('fall_detection')
    new_entry_ref = users_ref.push({
        'date': datetime.now().date().isoformat(),
        'time': datetime.now().time().isoformat(),
        'image': 'download_url'  # Save the download URL to the database
    })

    # Push data to Firebase
    # new_prediction.set(prediction_data)
    print("Prediction sent to Firebase")

except Exception as e:
    print(f"An error occurred: {e}")
