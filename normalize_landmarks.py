import pandas as pd
import numpy as np


INPUT_FILE = "data/train_landmarks.csv"
OUTPUT_FILE = "data/train_normalized.csv"


df = pd.read_csv(INPUT_FILE)


feature_columns = []

for i in range(21):
    feature_columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


normalized_data = []


for _, row in df.iterrows():

    landmarks = []

    wrist_x = row["x0"]
    wrist_y = row["y0"]
    wrist_z = row["z0"]


    for i in range(21):

        x = row[f"x{i}"]
        y = row[f"y{i}"]
        z = row[f"z{i}"]


        normalized_x = x - wrist_x
        normalized_y = y - wrist_y
        normalized_z = z - wrist_z


        landmarks.extend([
            normalized_x,
            normalized_y,
            normalized_z
        ])


    normalized_data.append(
        [row["label"]] + landmarks
    )


columns = ["label"] + feature_columns


normalized_df = pd.DataFrame(
    normalized_data,
    columns=columns
)


normalized_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("--------------------------------")
print("Normalization complete")
print("--------------------------------")
print("Samples:", len(normalized_df))
print("Features:", len(feature_columns))
print("Output:", OUTPUT_FILE)