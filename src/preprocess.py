import cv2
import os


IMG_SIZE = 224


def extract_face(image_path):
    """
    Detect the largest face in an image and return
    a resized face image.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read the image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    # Select the largest detected face
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    face = image[y:y+h, x:x+w]

    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

    return face


def save_face(image_path, output_path):
    """Extract and save a detected face."""

    face = extract_face(image_path)

    if face is None:
        print("No face detected:", image_path)
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, face)

    print("Saved:", output_path)
    return True


if __name__ == "__main__":
    print("Deepfake preprocessing module")
