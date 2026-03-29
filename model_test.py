import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Update paths to be relative to SCRIPT_DIR
MODEL_DIR = os.path.join(SCRIPT_DIR, "final")
AUDIO_PATH = os.path.join(SCRIPT_DIR, "audio_cap", "male_8_synth2.wav")
OUTPUT_TXT = os.path.join(SCRIPT_DIR, "output_rec_text", "transcription.txt")
ORIGINAL_MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-kannada-knm-560"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading processor from original model...")
processor = Wav2Vec2Processor.from_pretrained(ORIGINAL_MODEL_NAME)

print("Loading fine-tuned model...")
print(f"Model path: {MODEL_DIR}")
model = Wav2Vec2ForCTC.from_pretrained(MODEL_DIR).to(device)
model.eval()

print("Loading audio with soundfile...")
print(f"Audio path: {AUDIO_PATH}")
# Read audio file (soundfile returns samples as float64 in range [-1, 1])
audio, sr = sf.read(AUDIO_PATH)

# If stereo, convert to mono by averaging channels
if len(audio.shape) > 1:
    audio = audio.mean(axis=1)

# Resample if necessary (soundfile doesn't resample, so we use simple check)
if sr != 16000:
    print(f"Warning: sample rate is {sr}, but model expects 16000. Resampling not implemented in this script.")
    # For simplicity, we assume the file is already 16 kHz. If not, you'd need a resampler (e.g., librosa).
    # You can use torchaudio's resample or librosa, but to avoid dependencies we assume correct rate.
    # If your test file isn't 16 kHz, consider converting it beforehand.

# Convert to tensor and add batch dimension
input_values = processor(audio, sampling_rate=sr, return_tensors="pt").input_values.to(device)

with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print("Transcription:", transcription)

os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(transcription)

print(f"Saved to {OUTPUT_TXT}")