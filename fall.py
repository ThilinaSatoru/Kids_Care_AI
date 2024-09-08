import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

from configs.config import *

# Define a boolean flag for video mode
print('If Sample Video - 1, Camera - 2: ')
x = input()
if int(x) == 1:
    USE_VIDEO_FILE = True  # Set to True to use video file, False to use PiCamera
else:
    USE_VIDEO_FILE = False

# Initialize YOLO model
MODEL = YOLO(YOLO_MODEL_PATH)

# Firebase Configuration
users_ref = firebase_ref.child('fall_detection')

# Initialize PiCamera2 or Video Capture
if USE_VIDEO_FILE:
    # Open video file
    print("Using video file")
    video_capture = cv2.VideoCapture('samples/v1.mp4')
    if not video_capture.isOpened():
        raise IOError("Cannot open video file")
else:
    # Initialize PiCamera2
    print("Waiting for video")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (VIDEO_HEIGHT, VIDEO_WIDTH), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

try:
    while True:
        if USE_VIDEO_FILE:
            # Capture frame from video file
            ret, frame = video_capture.read()
            if not ret:
                print("End of video file or failed to read frame")
                break
        else:
            # Capture frame from PiCamera2
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if detected_fall:
            # Save the annotated frame if a fall is detected
            filename = os.path.join(IMG_OUTPUT_DIRECTORY, f"fall_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Detected Fall and saved frame to {filename}")

            # Upload the image to Firebase Storage
            blob = firebase_bucket.blob(f'falls/fall_{timestamp}.jpg')
            blob.upload_from_filename(filename)
            download_url = blob.public_url  # Get the image's download URL

            print(download_url)

            # Push data to Firebase Realtime Database
            new_entry_ref = users_ref.push({
                'date': datetime.now().date().isoformat(),
                'time': datetime.now().time().isoformat(),
                'image': download_url  # Save the download URL to the database
            })

        else:
            cv2.imwrite(f"captured_images/frame_{timestamp}.jpg", frame)
            print("No fall detected in this frame.")

except KeyboardInterrupt:
    print("Closing Camera Detection.")
finally:
    # Clean up
    if USE_VIDEO_FILE:
        video_capture.release()
    else:
        picam2.stop()
    cv2.destroyAllWindows()
