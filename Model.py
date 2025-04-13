from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import sys
import os
from flask import Flask, request, jsonify


def load_model():
    model_name = "openai/whisper-small" # use if model is not saved
    # model_name = "./whisper_model"    # use if model is saved
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Save model to local machine
    # model.save_pretrained("./whisper_model")
    # processor.save_pretrained("./whisper_model")
    return model, processor


class SpeechRecognizer:
    def __init__(self):
        self.model, self.processor = load_model()
        # self.path = path
        self.sample_rate = 16000

    def transcribe_audio(self, path):
        # Load audio file
        speech, rate = librosa.load(path=path, sr=self.sample_rate)

        # Process the audio
        input_values = self.processor(speech, return_tensors="pt", sampling_rate=self.sample_rate, task="translate", language="en").input_features

        # Perform inference
        logits = self.model.generate(input_values)

        # Decode prediction
        transcription = self.processor.batch_decode(logits, skip_special_tokens=True)[0]

        return transcription


app = Flask(__name__)

recognizer = SpeechRecognizer()


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = "temp.wav"
    file.save(file_path)

    transcript = recognizer.transcribe_audio(file_path)
    os.remove(file_path)

    print(transcript)
    return jsonify({"transcript": transcript})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
