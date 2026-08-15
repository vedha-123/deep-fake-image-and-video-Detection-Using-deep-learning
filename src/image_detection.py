import cv2
import numpy as np
import tensorflow as tf


IMG_SIZE = 224


def detect_deepfake(image_path, model_path):
    model = tf.keras.models.load_model(model_path)

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "FAKE"
        confidence = prediction
    else:
        result = "REAL"
        confidence = 1 - prediction

    return result, float(confidence)


if __name__ == "__main__":
    print("Deepfake image detection module")
