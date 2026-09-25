import cv2
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"
CLASSIFIER_PATH = "models/sign_model.joblib"


model = joblib.load(CLASSIFIER_PATH)


base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)


camera = cv2.VideoCapture(0)

frame_timestamp = 0


while True:

    success, frame = camera.read()

    if not success:
        print("Could not access camera")
        break


    frame = cv2.flip(frame, 1)


    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    frame_timestamp += 1


    results = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )


    if results.hand_landmarks:

        hand = results.hand_landmarks[0]


        wrist_x = hand[0].x
        wrist_y = hand[0].y
        wrist_z = hand[0].z


        features = []


        for landmark in hand:

            features.extend([
                landmark.x - wrist_x,
                landmark.y - wrist_y,
                landmark.z - wrist_z
            ])


        prediction = model.predict(
            [features]
        )[0]


        probabilities = model.predict_proba(
            [features]
        )[0]


        confidence = max(probabilities) * 100


        for landmark in hand:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        cv2.putText(
            frame,
            f"Sign: {prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )


        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "Show your hand",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    cv2.imshow(
        "AI Sign Language Recognition",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
detector.close()
cv2.destroyAllWindows()