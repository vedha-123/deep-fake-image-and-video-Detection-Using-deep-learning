import cv2
import numpy as np
import tensorflow as tf


IMG_SIZE = 224


def detect_video(video_path, model_path):
    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    predictions = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        frame = frame.astype("float32") / 255.0
        frame = np.expand_dims(frame, axis=0)

        prediction = model.predict(frame, verbose=0)[0][0]

        predictions.append(prediction)

    cap.release()

    if len(predictions) == 0:
        raise ValueError("No frames found in video.")

    average_prediction = np.mean(predictions)

    if average_prediction >= 0.5:
        result = "FAKE"
        confidence = average_prediction
    else:
        result = "REAL"
        confidence = 1 - average_prediction

    return result, float(confidence)


if __name__ == "__main__":
    print("Deepfake video detection module")
