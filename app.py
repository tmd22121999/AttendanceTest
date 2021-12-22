from sys import stdout
from makeup_artist import Makeup_artist
import logging
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from camera import Camera
from utils import base64_to_pil_image, pil_image_to_base64
import numpy as np
import cv2
from PIL import Image 

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app)
camera = Camera(Makeup_artist())


@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    camera.enqueue_input(input)
    image_data = processsss(input) # Do your magical Image processing here!!
    #image_data = image_data.decode("utf-8")
    image_data = "data:image/jpeg;base64," + image_data
    #print("OUTPUT " + image_data)
    emit('out-image-event', {'image_data': image_data}, namespace='/test')

def processsss(input2):
    img2 = base64_to_pil_image(input2)
    img2 = np.array(img2)
    cv2.putText(img2, "job: " , (25 + 6, 40 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0,255), 1)
    frame = Image.fromarray(img2)
    frame = frame.transpose(Image.FLIP_LEFT_RIGHT)
    frame = pil_image_to_base64(frame)
    
    return binascii.a2b_base64(frame)

@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    """Video streaming generator function."""

    app.logger.info("starting to generate frames!")
    while True:
        frame = camera.get_frame() #pil_image_to_base64(camera.get_frame())
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app)
