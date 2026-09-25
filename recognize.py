import cv2
import joblib
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_FILE = "models/sign_model_v2.joblib"
HAND_MODEL = "models/hand_landmarker.task"


# Load trained model
model = joblib.load(MODEL_FILE)


# MediaPipe configuration
base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.HandLandmarker.create_from_options(options)


def normalize_hand(hand):
    hand_array = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32
    )

    # Wrist = landmark 0
    wrist = hand_array[0].copy()

    # Move wrist to origin
    hand_array = hand_array - wrist

    # Scale using maximum distance from wrist
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


# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


timestamp = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 1

    results = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    hand_count = len(results.hand_landmarks)

    # Default values
    prediction_text = "No hand detected"

    top_predictions = [
        ("-", 0.0),
        ("-", 0.0),
        ("-", 0.0)
    ]

    if hand_count > 0:

        features = create_features(results)

        features = features.reshape(1, -1)

        # Get probabilities
        probabilities = model.predict_proba(features)[0]

        # Top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]

        top_predictions = [
            (
                model.classes_[i],
                probabilities[i] * 100
            )
            for i in top_indices
        ]

        prediction = top_predictions[0][0]
        confidence = top_predictions[0][1]

        prediction_text = f"{prediction} ({confidence:.1f}%)"


    # -----------------------------
    # Display prediction
    # -----------------------------

    cv2.putText(
        frame,
        f"Prediction: {prediction_text}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"2nd: {top_predictions[1][0]} ({top_predictions[1][1]:.1f}%)",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"3rd: {top_predictions[2][0]} ({top_predictions[2][1]:.1f}%)",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Hands detected: {hand_count}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "ISL Sign Recognition - Model V2",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()