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
import binascii
import base64
import face_recognition
from io import BytesIO

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app)
camera = Camera(Makeup_artist())
known_face_encodings = []
known_face_names = []

print("Encoding face...")
FACE_IMG_DIR = os.path.join(os.getcwd(), 'face_img')
for filename in os.listdir(FACE_IMG_DIR):
    face = face_recognition.load_image_file(
        os.path.join(FACE_IMG_DIR, filename))
    face_encoding = face_recognition.face_encodings(face)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(re.sub('.jpg$', '', filename))
print("Encoding face done")
print(known_face_names)

# Prepare attendece set which contains the names of those who attended
attendence_set = set()

@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    camera.enqueue_input(input)
    image_data = input
    image_data = recog(input) # Do your magical Image processing here!!
    buf = BytesIO()
    image_data.save(buf, format="JPEG")
    image_data = base64.b64encode(buf.getvalue())
    image_data = image_data.decode("utf-8")
    image_data = "data:image/jpeg;base64," + image_data
    #print("OUTPUT " + image_data)
    emit('out-image-event', {'image_data': image_data}, namespace='/test')
    
    
    
def recog(img):
        frame =  np.array(img.convert('RGB') )
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
                print(face_names)

        process_this_frame = not process_this_frame
        # process_this_frame = False

        # Add person to attendence set
        for name in face_names:
            attendence_set.add(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, "wnnvownoveusjwovnwweqweqwe", (25 + 6, 40 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
        # Display the resulting image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = Image.fromarray(frame)
        return frame.transpose(Image.FLIP_LEFT_RIGHT)
    
    
def processsss(input2):
    img2 = base64_to_pil_image(input2)
    img2 = np.array(img2)
    cv2.putText(img2, "job: " , (25 + 6, 40 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0,255), 1)
    frame = Image.fromarray(img2)
    frame = frame.transpose(Image.FLIP_LEFT_RIGHT)
    
    return frame

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
