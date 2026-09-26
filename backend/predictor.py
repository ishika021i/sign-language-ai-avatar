import os
import cv2
import joblib
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "sign_model_v2.joblib"
)

HAND_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "hand_landmarker.task"
)


model = joblib.load(MODEL_FILE)


base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


def normalize_hand(hand):

    hand_array = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32
    )

    wrist = hand_array[0].copy()

    hand_array = hand_array - wrist

    distances = np.linalg.norm(
        hand_array,
        axis=1
    )

    scale = np.max(distances)

    if scale == 0:
        scale = 1

    hand_array = hand_array / scale

    return hand_array.flatten()


def create_features(results):

    left_hand = np.zeros(
        63,
        dtype=np.float32
    )

    right_hand = np.zeros(
        63,
        dtype=np.float32
    )

    for index, hand in enumerate(
        results.hand_landmarks
    ):

        handedness = (
            results.handedness[index][0].category_name
        )

        normalized = normalize_hand(hand)

        if handedness == "Left":
            left_hand = normalized

        elif handedness == "Right":
            right_hand = normalized

    return np.concatenate([
        left_hand,
        right_hand
    ])


def predict_image(image):

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    results = landmarker.detect(mp_image)

    hand_count = len(
        results.hand_landmarks
    )

    if hand_count == 0:
        return {
            "success": False,
            "message": "No hand detected"
        }

    features = create_features(results)

    # Prediction using normal Left/Right ordering
    normal_features = features.reshape(1, -1)
    normal_probabilities = model.predict_proba(normal_features)[0]

    # Prediction using swapped Left/Right ordering
    left_hand = features[:63]
    right_hand = features[63:]

    swapped_features = np.concatenate(
        [right_hand, left_hand]
    ).reshape(1, -1)

    swapped_probabilities = model.predict_proba(swapped_features)[0]

    # Choose the orientation with higher confidence
    if np.max(swapped_probabilities) > np.max(normal_probabilities):
        probabilities = swapped_probabilities
    else:
        probabilities = normal_probabilities

    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = [
        {
            "letter": str(model.classes_[i]),
            "confidence": round(
                float(probabilities[i] * 100),
                2
            )
        }
        for i in top_indices
    ]

    return {
        "success": True,
        "prediction": top_predictions[0]["letter"],
        "confidence": top_predictions[0]["confidence"],
        "hands_detected": hand_count,
        "top_predictions": top_predictions
    }