"""
predict.py
-----------
Image prediction module for MobileNetV2
"""

import time
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# -------------------------------------------------
# Load MobileNetV2 only once
# -------------------------------------------------
model = MobileNetV2(weights="imagenet")


def predict_image(image_file):
    """
    Predict the uploaded image.

    Parameters
    ----------
    image_file : Uploaded file or image path

    Returns
    -------
    list
        Top 5 ImageNet predictions
    """

    # Load image
    img = image.load_img(
        image_file,
        target_size=(224, 224)
    )

    # Convert to array
    img_array = image.img_to_array(img)

    # Create batch
    img_batch = np.expand_dims(img_array, axis=0)

    # Preprocess
    processed = preprocess_input(img_batch)

    # Prediction
    start = time.time()

    predictions = model.predict(
        processed,
        verbose=0
    )

    prediction_time = round(
        (time.time() - start) * 1000,
        2
    )

    decoded = decode_predictions(
        predictions,
        top=5
    )[0]

    print(f"Prediction Time: {prediction_time} ms")

    return decoded


def predict_from_path(image_path):
    """
    Predict image from local path.
    """

    return predict_image(image_path)


if __name__ == "__main__":

    path = "cat.png"

    results = predict_from_path(path)
    
    print("\nTop Predictions\n")

    for _, label, probability in results:

        print(
            f"{label:<25} {probability*100:.2f}%"
        )
        
        