import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

model_path = "./adapted_model_augmented_final"  # adjust path

print("Loading model...")
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)
model.eval()

audio_path = "audio_cap/72.wav"  # change to your test file
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

# Normalize
if audio.max() > 1.0:
    audio = audio / audio.max()

inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

with torch.no_grad():
    logits = model(inputs).logits
    ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(ids[0], skip_special_tokens=True)

print("Transcription:", transcription)