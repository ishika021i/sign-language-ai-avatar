import pandas as pd
import numpy as np

INPUT_FILE = "data/train_landmarks_v2.csv"
OUTPUT_FILE = "data/train_normalized_v2.csv"

df = pd.read_csv(INPUT_FILE)

labels = df["label"]
features = df.drop(columns=["label"])

X = features.values.astype(float)

# Left hand: columns 0-62
left = X[:, :63].reshape(-1, 21, 3)

# Right hand: columns 63-125
right = X[:, 63:126].reshape(-1, 21, 3)


def normalize_hand(hand):
    # Wrist is landmark 0
    wrist = hand[:, 0:1, :]

    # Move wrist to origin
    hand = hand - wrist

    # Scale based on maximum distance from wrist
    distances = np.linalg.norm(hand, axis=2)
    scale = np.max(distances, axis=1, keepdims=True)

    # Prevent division by zero
    scale[scale == 0] = 1

    hand = hand / scale[:, :, None]

    return hand


left_normalized = normalize_hand(left)
right_normalized = normalize_hand(right)

# Flatten back to 63 features per hand
left_flat = left_normalized.reshape(-1, 63)
right_flat = right_normalized.reshape(-1, 63)

normalized = np.hstack([left_flat, right_flat])

# Create output dataframe
feature_names = []

for hand_name in ["left", "right"]:
    for i in range(21):
        feature_names.extend([
            f"{hand_name}_x{i}",
            f"{hand_name}_y{i}",
            f"{hand_name}_z{i}"
        ])

normalized_df = pd.DataFrame(
    normalized,
    columns=feature_names
)

normalized_df.insert(0, "label", labels.values)

normalized_df.to_csv(OUTPUT_FILE, index=False)

print("--------------------------------")
print("Two-hand normalization complete")
print("--------------------------------")
print(f"Samples: {len(normalized_df)}")
print(f"Features: {len(feature_names)}")
print(f"Output: {OUTPUT_FILE}")