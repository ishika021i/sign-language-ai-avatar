import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_FILE = "models/sign_model_v2.joblib"
HAND_MODEL = "models/hand_landmarker.task"

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

landmarker = vision.HandLandmarker.create_from_options(options)


def normalize_hand(hand):

    hand_array = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32
    )

    wrist = hand_array[0].copy()

    hand_array = hand_array - wrist

    distances = np.linalg.norm(hand_array, axis=1)

    scale = np.max(distances)

    if scale == 0:
        scale = 1

    hand_array = hand_array / scale

    return hand_array.flatten()


def create_features(results):

    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    for index, hand in enumerate(results.hand_landmarks):

        handedness = results.handedness[index][0].category_name

        normalized = normalize_hand(hand)

        if handedness == "Left":
            left_hand = normalized

        elif handedness == "Right":
            right_hand = normalized

    return np.concatenate([
        left_hand,
        right_hand
    ])


import os
import glob

files = glob.glob("dataset/Training/S/*")

correct = 0
total = 0

print("--------------------------------")
print("Testing Model V2 on S images")
print("--------------------------------")

for file in files:

    image = cv2.imread(file)

    if image is None:
        continue

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    results = landmarker.detect(mp_image)

    if len(results.hand_landmarks) == 0:
        continue

    features = create_features(results)

    features = features.reshape(1, -1)

    probabilities = model.predict_proba(features)[0]

    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = [
        (
            model.classes_[i],
            probabilities[i] * 100
        )
        for i in top_indices
    ]

    prediction = top_predictions[0][0]

    total += 1

    if prediction == "S":
        correct += 1

    if total <= 10:

        print(
            os.path.basename(file),
            "→",
            top_predictions
        )


print("--------------------------------")
print(f"S images tested: {total}")
print(f"Correct S predictions: {correct}")

if total > 0:
    print(
        f"Accuracy on S images: "
        f"{correct / total * 100:.2f}%"
    )

print("--------------------------------")

landmarker.close()