import cv2
import random

def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    return face


def predict(image):
    face = extract_face(image)

    if face is None:
        return None, None

    # DEMO PROBABILITY (prototype)
    fake_probability = random.uniform(5, 95)

    return face, round(fake_probability, 2)
