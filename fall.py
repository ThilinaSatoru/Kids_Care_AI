import os

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

from configs.config import *

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (VIDEO_HEIGHT, VIDEO_WIDTH), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Initialize YOLO model
MODEL = YOLO(YOLO_MODEL_PATH)

# Firebase Configuration
users_ref = firebase_ref.child('fall_detection')

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()

        # Run YOLO detection
        results = MODEL.track(frame, persist=True, conf=0.5)

        detected_fall = False

        for obj in results[0].boxes:
            class_id = int(obj.cls)
            class_name = MODEL.names[class_id]
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
                cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            color, 2)

        timestamp = date_now.strftime("%Y%m%d_%H%M%S")
        if detected_fall:
            # Save the annotated frame if a fall is detected
            filename = os.path.join(IMG_OUTPUT_DIRECTORY, f"fall_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Detected Fall and saved frame to {filename}")

            # Upload the image to Firebase Storage
            blob = firebase_bucket.blob(f'falls/fall_{timestamp}.jpg')
            blob.upload_from_filename(filename)
            download_url = blob.public_url  # Get the image's download URL

            # Push data to Firebase Realtime Database
            new_entry_ref = users_ref.push({
                'date': date_now.date().isoformat(),
                'time': date_now.time().isoformat(),
                'image': download_url  # Save the download URL to the database
            })

        else:
            cv2.imwrite(f"captured_images/frame_{timestamp}.jpg", frame)
            print("No fall detected in this frame.")

except KeyboardInterrupt:
    print("Closing Camera Detection.")
finally:
    # Clean up
    cv2.destroyAllWindows()
    picam2.stop()
