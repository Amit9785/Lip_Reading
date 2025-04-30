from flask import Flask, render_template, request, url_for
import os
import numpy as np
import tensorflow as tf
import time
import shutil
import subprocess
from util import load_data, num_to_char
from modelutil import load_model

app = Flask(__name__)

# Paths
DATA_PATH = os.path.join('..', 'data', 's1')
STATIC_PATH = os.path.join('static')
OUTPUT_VIDEO = os.path.join(STATIC_PATH, 'test_video.mp4')


def decode_ctc_output(pred_tokens):
    text = []
    previous = None
    for token in pred_tokens:
        if token == -1 or token == 0:
            continue
        if token != previous:
            char = num_to_char(tf.constant([token])).numpy()[0]
            if isinstance(char, bytes):
                char = char.decode("utf-8")
            text.append(char)
        previous = token
    return ''.join(text)


def convert_video_to_mp4(input_path, output_path):
    """
    Convert video to H.264/AAC using ffmpeg.
    """
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_path,
            '-vcodec', 'libx264', '-acodec', 'aac',
            output_path
        ], check=True)
        print(f"Video converted and saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in ffmpeg: {e}")
        return False


@app.route('/')
def index():
    options = os.listdir(DATA_PATH)
    return render_template('index.html', options=options)


@app.route('/video', methods=['POST'])
def video():
    selected_video = request.form['selected_video']
    source_path = os.path.join(DATA_PATH, selected_video)

    if not os.path.exists(source_path):
        return f"Error: File {source_path} does not exist."

    print(f"Loading video: {source_path}")

    try:
        video_tensor, annotations = load_data(tf.convert_to_tensor(source_path))
    except Exception as e:
        return f"Error loading video: {str(e)}"

    if video_tensor is None:
        return "Error: Failed to load video frames."

    try:
        model = load_model()
    except Exception as e:
        return f"Error loading model: {str(e)}"

    if model is None:
        return "Error: Failed to load the model."

    print("Model loaded successfully.")

    try:
        yhat = model.predict(tf.expand_dims(video_tensor, axis=0))
        decoded = tf.argmax(yhat, axis=-1)[0].numpy().tolist()
        cleaned_text = decode_ctc_output(decoded)
        softmax_summary = np.round(yhat[0][:10], 3).tolist()
        token_ids = ' '.join(str(x) for x in decoded)

        # Convert/copy the video to the static folder
        if not convert_video_to_mp4(source_path, OUTPUT_VIDEO):
            shutil.copy(source_path, OUTPUT_VIDEO)

        return render_template('video_result.html',
                               selected_video=selected_video,
                               softmax_summary=softmax_summary,
                               model_output=token_ids,
                               decoded_text=cleaned_text,
                               time=int(time.time()))

    except Exception as e:
        return f"Error during prediction or decoding: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
