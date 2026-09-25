import os
import csv
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"
DATASET_PATH = "dataset/Training"
OUTPUT_FILE = "data/train_landmarks_v2.csv"


os.makedirs("data", exist_ok=True)


base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)


detector = vision.HandLandmarker.create_from_options(
    options
)


header = ["label"]


for hand_name in ["left", "right"]:

    for i in range(21):

        header.extend([
            f"{hand_name}_x{i}",
            f"{hand_name}_y{i}",
            f"{hand_name}_z{i}"
        ])


with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(header)

    total_images = 0
    detected_images = 0
    two_hand_images = 0


    for label in sorted(
        os.listdir(DATASET_PATH)
    ):

        label_path = os.path.join(
            DATASET_PATH,
            label
        )


        if not os.path.isdir(label_path):
            continue


        print(f"\nProcessing sign: {label}")


        for filename in os.listdir(label_path):

            image_path = os.path.join(
                label_path,
                filename
            )


            image = cv2.imread(
                image_path
            )


            if image is None:
                continue


            total_images += 1


            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )


            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )


            results = detector.detect(
                mp_image
            )


            if not results.hand_landmarks:
                continue


            detected_images += 1


            left_hand = None
            right_hand = None


            for index, hand in enumerate(
                results.hand_landmarks
            ):

                handedness = (
                    results.handedness[index][0].category_name
                )


                if handedness == "Left":

                    left_hand = hand

                elif handedness == "Right":

                    right_hand = hand


            if (
                left_hand is not None
                and right_hand is not None
            ):

                two_hand_images += 1


            row = [label]


            for hand in [
                left_hand,
                right_hand
            ]:

                if hand is None:

                    row.extend(
                        [0.0] * 63
                    )

                else:

                    for landmark in hand:

                        row.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])


            writer.writerow(row)


        print(
            f"Finished {label}"
        )


detector.close()


print("\n--------------------------------")
print("Two-hand extraction complete")
print("--------------------------------")

print(
    f"Total images: {total_images}"
)

print(
    f"Images with hands: {detected_images}"
)

print(
    f"Images with two hands: {two_hand_images}"
)

print(
    f"Output: {OUTPUT_FILE}"
)