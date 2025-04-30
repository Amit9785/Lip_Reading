import tensorflow as tf
import cv2
import os
from typing import List

# Define the vocabulary
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Char-to-int and int-to-char mappings
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Failed to read a frame from {path}. Skipping frame.")
            continue

        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[170:240, 60:240, :])

    cap.release()

    if not frames:
        return None  # Handle empty or unreadable videos

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_data(path: str):
    path = bytes.decode(path.numpy()) if isinstance(path, tf.Tensor) else path
    file_name = os.path.splitext(os.path.basename(path))[0]
    video_path = os.path.join('..', 'data', 's1', f'{file_name}.mpg')
    frames = load_video(video_path)
    return frames, None  # Returning a tuple with frames and a None placeholder for annotations

