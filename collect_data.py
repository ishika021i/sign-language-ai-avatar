import cv2
import csv
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"
DATA_FILE = "data/sign_landmarks.csv"


os.makedirs("data", exist_ok=True)


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

detector = vision.HandLandmarker.create_from_options(options)


if not os.path.exists(DATA_FILE):

    with open(DATA_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        header = ["label"]

        for i in range(21):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        writer.writerow(header)


signs = [
    "HELLO",
    "YES",
    "NO",
    "THANK_YOU",
    "HELP"
]


print("\nSign Language Dataset Collector")
print("--------------------------------")

for index, sign in enumerate(signs, start=1):
    print(f"{index}. {sign}")

choice = int(input("\nEnter sign number: "))

label = signs[choice - 1]

print(f"\nSelected sign: {label}")
print("Show the sign to the camera.")
print("Press SPACE to capture a sample.")
print("Press Q to quit.\n")


camera = cv2.VideoCapture(0)

frame_timestamp = 0
sample_count = 0


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
            f"Sign: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "Show your hand",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    cv2.imshow(
        "ISL Dataset Collector",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


    if key == 32 and results.hand_landmarks:

        hand = results.hand_landmarks[0]

        row = [label]

        for landmark in hand:

            row.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])


        with open(
            DATA_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)
            writer.writerow(row)


        sample_count += 1

        print(
            f"{label}: sample {sample_count} saved"
        )


camera.release()
detector.close()
cv2.destroyAllWindows()

print("\nCollection finished.")
print(f"Samples collected for {label}: {sample_count}")