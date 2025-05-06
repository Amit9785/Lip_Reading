# 🧠 LipBuddy – Deep Learning-based Lip Reading App

**LipBuddy** is a web application that performs **lip reading from videos** using the **LipNet deep learning model**. It allows users to upload or select a video and then extracts and decodes speech based purely on lip movement, without audio.

---

## 📌 Features

- 🎥 Upload or select a video for lip reading.
- 🧠 Processes the video using the **LipNet** model.
- 📜 Displays both decoded text (speech) and raw model outputs (tokens/logits).s
- 🌐 Web interface available in **Flask** and **Streamlit** versions.
- 📁 Uses a simplified video input directory: `data/s1`.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Amit9786/LipBuddy.git
cd LipBuddy



### 2. Install dependencies
pip install -r requirements.txt


### 3. Download LipNet model weights
Ensure you have the pretrained LipNet weights saved in the correct path (model/ or as required by your implementation).


### 4. Run the App
For Flask version:

python app.py

### 5. Access the app

    Flask: http://localhost:5000

    Streamlit: Provided in terminal after launch


🧬 Model Details

LipBuddy uses the LipNet model, which includes:

    Spatiotemporal Convolutional Neural Networks (ST-CNNs)

    Bidirectional LSTM layers

    Connectionist Temporal Classification (CTC) decoding

Original Paper:
Assael et al., 2016 – "LipNet: End-to-End Sentence-Level Lipreading"



### 💡 Use Cases

    Assistive tech for the hearing-impaired

    Silent communication systems

    Security and surveillance

    Silent speech interfaces in noisy environments


