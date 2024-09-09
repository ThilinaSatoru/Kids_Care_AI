import io
import os

import libcamera
from PIL import Image
from flask import Flask, render_template, Response
from picamera2 import Picamera2

app = Flask(__name__)

camera = Picamera2()
camera_config = camera.create_video_configuration(main={"size": (640, 480)},
                                                  transform=libcamera.Transform(hflip=True, vflip=True))
camera.configure(camera_config)
camera.start()


# Function to get system information
def get_system_info():
    cpu_temp = os.popen("vcgencmd measure_temp").readline().replace("temp=", "").strip()
    cpu_usage = os.popen("top -bn1 | grep load | awk '{printf \"%.2f\", $(NF-2)}'").readline().strip()
    mem_usage = os.popen("free -m | awk 'NR==2{printf \"%s/%sMB (%.2f%%)\", $3,$2,$3*100/$2 }'").readline().strip()
    disk_usage = os.popen("df -h | awk '$NF==\"/\"{printf \"%d/%dGB (%s)\", $3,$2,$5}'").readline().strip()
    return {"cpu_temp": cpu_temp, "cpu_usage": cpu_usage, "mem_usage": mem_usage, "disk_usage": disk_usage}


# Route for the home page
@app.route('/')
def index():
    sys_info = get_system_info()
    return render_template('index.html', sys_info=sys_info)


def generate_stream():
    stream = io.BytesIO()
    while True:
        frame = camera.capture_array()  # Capture image as a NumPy array
        img = Image.fromarray(frame)  # Convert the array to a PIL Image object

        if img.mode == 'RGBA':
            img = img.convert('RGB')  # Convert RGBA to RGB

        stream.seek(0)
        img.save(stream, format='JPEG')  # Save the image as JPEG in the stream
        stream.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + stream.read() + b'\r\n')
        stream.seek(0)
        stream.truncate()


# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
