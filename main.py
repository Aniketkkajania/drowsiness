from flask import Flask, request, jsonify
import cv2
import numpy as np
import dlib
from imutils import face_utils

app = Flask(__name__)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

def detect_status(image_path):
    global sleep, drowsy, active # Declare these variables as global

    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    status = ""

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)


        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])


        if left_blink == 0 or right_blink == 0:
            status = "SLEEPING !!!"
        elif left_blink == 1 or right_blink == 1:
            status = "Drowsy !"
        else:
            status = "Active :)"

    return status

@app.route('/detect_status', methods=['POST'])
def detect_status_api():
    if 'file' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['file']
    image_path = "uploaded_image.jpg"  # Save the image temporarily

    image_file.save(image_path)
    status = detect_status(image_path)

    return jsonify({"status": status})

