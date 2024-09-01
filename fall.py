import cv2
from ultralytics import YOLO
from datetime import datetime
from picamera2 import Picamera2
import time
import os
import firebase_admin
from firebase_admin import credentials, storage, db

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Initialize YOLO model
model = YOLO('/fall-ml/falldetectionmodelv3.pt')
now = datetime.now()

# Initialize Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("kidzcare-97f3c-firebase-adminsdk-fpml6-3904295b4d.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'kidzcare-97f3c.appspot.com ',
        'databaseURL': 'https://kidzcare-97f3c-default-rtdb.firebaseio.com'
    })
# Initialize Firebase Storage bucket
bucket = storage.bucket()  
ref = db.reference('/kiddycare')
users_ref = ref.child('fall_detection')

# Create directories for saving images
output_dir = "detected_falls"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("captured_images", exist_ok=True)

# Initialize variables
frame_count = 0
save_interval = 30  # Save a non-fall image every 30 frames

while True:
    # Capture frame
    frame = picam2.capture_array()
    
    # Increment frame count
    frame_count += 1
    
    # Run YOLO detection
    results = model.track(frame, persist=True, conf=0.5)
    
    detected_fall = False

    for obj in results[0].boxes:
        class_id = int(obj.cls)
        class_name = model.names[class_id]
        bbox = obj.xyxy[0]
        confidence = float(obj.conf)
        x1, y1, x2, y2 = bbox.int().tolist()

        if confidence > 0.8:
            if class_name == 'Fall':
                color = (0, 0, 255)  # Red color for 'Fall'
                detected_fall = True
            else:
                color = (0, 255, 0)  # Green color for 'Not Fall'

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if detected_fall:
        # Save the annotated frame if a fall is detected
        filename = os.path.join(output_dir, f"fall_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Detected Fall and saved frame to {filename}")

        # Upload the image to Firebase Storage
        blob = bucket.blob(f'falls/fall_{timestamp}.jpg')
        blob.upload_from_filename(filename)
        download_url = blob.public_url  # Get the image's download URL

        # Push data to Firebase Realtime Database
        new_entry_ref = users_ref.push({
            'date': now.date().isoformat(),
            'time': now.time().isoformat(),
            'image': download_url  # Save the download URL to the database
        })

    else:
        cv2.imwrite(f"captured_images/frame_{timestamp}.jpg", frame)
        print("No fall detected in this frame.")

    # Clean up
    cv2.destroyAllWindows()
    picam2.stop()
