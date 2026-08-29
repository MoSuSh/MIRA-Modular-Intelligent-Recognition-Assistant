#!/usr/bin/env python3
# ============================================================
# MIRA UNIFIED - FINAL
# ============================================================

import os
import cv2
import time
import numpy as np
import re
from datetime import datetime
import subprocess
import json
import queue
import sounddevice as sd
from scipy import signal
from ultralytics import YOLO
import torch
import timm
from vosk import Model as VoskModel, KaldiRecognizer
from rapidocr_onnxruntime import RapidOCR
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import threading
import soundfile as sf
import requests

# ---------- Paths ----------
# ---------- Paths ----------
API_BASE_URL = "http://localhost:8000"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(SCRIPT_DIR, "captures")
MEMORY_DIR = os.path.join(SCRIPT_DIR, "mira_memory")
VOSK_MODEL_PATH = os.path.join(SCRIPT_DIR, "vosk-model-small-en-in-0.4")
YOLO_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolov8n.pt")

DET_MODEL_PATH = os.path.join(SCRIPT_DIR, "mira_models", "detection", "v5", "det.onnx")
REC_MODEL_PATH = os.path.join(SCRIPT_DIR, "mira_models", "languages", "english", "rec.onnx")
DICT_PATH = os.path.join(SCRIPT_DIR, "mira_models", "languages", "english", "dict.txt")

KANNADA_MODEL_DIR = os.path.join(SCRIPT_DIR, "final_model")
ORIGINAL_MODEL_NAME = "Harveenchadha/vakyansh-wav2vec2-kannada-knm-560"

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

# ---------- Global Settings ----------
FRAME_W, FRAME_H = 640, 480
MIN_STABLE_SECONDS = 1.5
PRE_CAPTURE_DELAY = 0.8
BURST_COUNT = 6
BURST_INTERVAL = 0.07
SHARPNESS_WEIGHT = 0.7
UPSCALE_FACTOR = 1.1
OCR_COOLDOWN = 3.0
DISPLAY_OCR_RESULT_SEC = 6
OCCLUSION_TRIGGER_SEC = 3.0
DARK_MEAN_THRESHOLD = 25
MEM_SIM_THRESHOLD = 0.70
HUMAN_CLASSES = {"person","man","woman","boy","girl","face","human","head","body"}
COLOR_TARGET_CLASSES = {"bottle","cup","bowl","plate","knife","fork","spoon",
    "shirt","pants","backpack","handbag","umbrella","laptop","cell phone",
    "remote","keyboard","mouse","book","banana","apple","orange","carrot"}
COLOR_SPEAK_COOLDOWN = 3.0
COLOR_MIN_CONFIDENCE = 0.15
TTS_ENABLED = True

YOLO_EVERY_N_FRAMES = 5
frame_counter = 0
last_boxes = None
last_names = None
last_annotated = None

# ---------- Serialised TTS ----------
tts_lock = threading.Lock()
tts_queue = queue.Queue()
tts_worker_thread = None

def _tts_worker():
    while True:
        func, args, kwargs = tts_queue.get()
        if func is None:
            break
        with tts_lock:
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"TTS error: {e}")
        tts_queue.task_done()

def _speak_text_internal(text):
    if not TTS_ENABLED:
        return
    clean = re.sub(r'[^\w\s]', '', text)
    subprocess.run(['espeak', clean], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _speak_kannada_internal(text):
    if not TTS_ENABLED:
        return
    try:
        subprocess.run(['espeak', '-v', 'kn', text], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Kannada TTS error: {e}")

def speak_text(text):
    tts_queue.put((_speak_text_internal, (text,), {}))

def speak_kannada(text):
    tts_queue.put((_speak_kannada_internal, (text,), {}))

# ---------- Camera handling ----------
cap = None

def init_camera():
    global cap
    if cap is not None:
        cap.release()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 10)
    print(f"Camera initialized: {int(cap.get(3))}x{int(cap.get(4))}")

def release_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        print("Camera released (power saving)")

def ensure_camera():
    global cap
    if cap is None:
        init_camera()
    return cap

def set_camera_resolution(width, height, device='/dev/video0'):
    try:
        subprocess.run(['v4l2-ctl', '-d', device, f'--set-fmt-video=width={width},height={height}'],
                       check=True, capture_output=True)
        print(f"Camera resolution set to {width}x{height}")
    except:
        pass

def optimize_camera_for_ocr(device='/dev/video0'):
    try:
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'auto_exposure=1'], check=True, capture_output=True)
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'white_balance_automatic=1'], check=True, capture_output=True)
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'sharpness=6'], check=True, capture_output=True)
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'focus_automatic_continuous=0'], check=True, capture_output=True)
        time.sleep(0.05)
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'focus_automatic_continuous=1'], check=True, capture_output=True)
        print("Camera optimized for OCR")
    except:
        pass

def refocus_camera(device='/dev/video0'):
    try:
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'focus_automatic_continuous=0'], check=True, capture_output=True)
        time.sleep(0.05)
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'focus_automatic_continuous=1'], check=True, capture_output=True)
        print("Camera refocus triggered")
    except:
        pass

set_camera_resolution(FRAME_W, FRAME_H)
init_camera()

# ---------- Sharpness & Preprocessing ----------
def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def preprocess(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    m = np.mean(l)
    gamma = 1.8 if m < 60 else 1.4 if m < 100 else 0.95 if m > 200 else 1.0
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    l = cv2.LUT(l, table)
    clahe = cv2.createCLAHE(1.1, (8, 8))
    l = clahe.apply(l)
    l = cv2.fastNlMeansDenoising(l, None, 3, 7, 21)
    blur = cv2.GaussianBlur(l, (3, 3), 0)
    l = cv2.addWeighted(l, 1.05, blur, -0.05, 0)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if UPSCALE_FACTOR != 1:
        result = cv2.resize(result, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
                            interpolation=cv2.INTER_CUBIC)
    return result

def clean_text(t):
    t = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', t)
    t = re.sub(r'[^A-Za-z0-9.,;:!?\'"()\-\s]', ' ', t)
    return re.sub(r'\s{2,}', ' ', t).strip()

def save_result(raw, proc, lines):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(CAPTURES_DIR, f"cap_{ts}.jpg"), raw)
    cv2.imwrite(os.path.join(CAPTURES_DIR, f"cap_{ts}_proc.jpg"), proc)
    with open(os.path.join(CAPTURES_DIR, f"cap_{ts}.txt"), "w", encoding="utf-8") as f:
        for L in lines:
            f.write(L + "\n")

# ---------- Memory ----------
print("Loading EfficientNet-lite0...")
embed_model = timm.create_model("tf_efficientnet_lite0", pretrained=True, num_classes=0, global_pool="avg")
embed_model.eval()

def extract_embedding(img):
    img_resized = cv2.resize(img, (224,224))
    img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0)/255.0
    with torch.no_grad():
        return embed_model(img_tensor).numpy().flatten().tolist()

def save_memory(name, embedding):
    url = f"{API_BASE_URL}/memories"
    payload = {
        "name": name,
        "embedding": embedding
    }
    try:
        response = requests.post(url, json=payload, timeout=3.0)
        if response.status_code in (200,201) :
            print(f"Permanently saved to SQLite database via API: {name}")
        else:
            print(f"API rejected memory storage: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Backend server unreachable. Memory for '{name}' NOT saved: {e}")

def load_memories():
    url = f"{API_BASE_URL}/memories"
    try:
        response = requests.get(url, timeout=3.0)
        if response.status_code in (200,201) :
            return response.json()
        else:
            print(f"Failed to retrieve memories from API: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Backend server unreachable during memory retrieval: {e}")
        return []

def cos_sim(a,b):
    a = np.array(a); b = np.array(b)
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denom) if denom != 0 else 0.0

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA)*max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter/(areaA+areaB-inter+1e-6)

# ---------- Color detection ----------
COLOR_RANGES = {
    "red":    [([0,50,50],[10,255,255]), ([170,50,50],[180,255,255])],
    "green":  [([40,40,40],[80,255,255])],
    "blue":   [([100,50,50],[130,255,255])],
    "yellow": [([20,50,50],[35,255,255])],
    "orange": [([10,50,50],[20,255,255])],
    "purple": [([130,50,50],[160,255,255])],
    "pink":   [([160,50,50],[170,255,255])],
    "brown":  [([10,50,20],[20,255,150])],
    "gray":   [([0,0,40],[180,40,220])],
    "black":  [([0,0,0],[180,255,50])],
    "white":  [([0,0,220],[180,40,255])]
}

def get_dominant_color_simple(crop):
    if crop.size==0: return "none",0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total = crop.shape[0]*crop.shape[1]
    best_color, best_count = "unknown", 0
    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (low,high) in ranges:
            mask += cv2.inRange(hsv, np.array(low), np.array(high))
        cnt = cv2.countNonZero(mask)
        if cnt > best_count:
            best_count, best_color = cnt, color
    return best_color, best_count/total

# ---------- English Voice (Vosk) ----------
if os.path.exists(VOSK_MODEL_PATH):
    print(f"Loading Vosk from {VOSK_MODEL_PATH}")
    vosk_model = VoskModel(VOSK_MODEL_PATH)
else:
    print("Vosk model not found")
    vosk_model = None

def listen_once(prompt=None, timeout=8.0):
    if vosk_model is None:
        return ""
    if prompt:
        speak_text(prompt)

    DEVICE_INDEX = 1          # <-- CHANGED to device 1
    NATIVE_RATE = 44100
    TARGET_RATE = 16000
    BLOCKSIZE = int(NATIVE_RATE * 0.1)

    audio_queue = queue.Queue(maxsize=50)
    def callback(indata, frames, time_info, status):
        try:
            audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

    rec = KaldiRecognizer(vosk_model, TARGET_RATE)
    try:
        with sd.InputStream(
            device=DEVICE_INDEX,
            samplerate=NATIVE_RATE,
            blocksize=BLOCKSIZE,
            dtype='int16',
            channels=1,
            callback=callback
        ):
            print("Listening...")
            start = time.time()
            chunks = []
            while time.time() - start < timeout:
                try:
                    data = audio_queue.get(timeout=0.2)
                    chunks.append(data)
                except queue.Empty:
                    continue
            if not chunks:
                return ""
            audio = np.concatenate(chunks, axis=0).flatten()
            # Resample to 16 kHz
            num_samples = int(len(audio) * TARGET_RATE / NATIVE_RATE)
            resampled = signal.resample(audio, num_samples).astype(np.int16)
            rec.AcceptWaveform(resampled.tobytes())
            result = json.loads(rec.FinalResult())
            text = result.get("text", "").strip()
            print(f"🗣️ USER SAID: {text}")
            return text
    except Exception as e:
        print(f"Voice capture error: {e}")
        return ""

# ---------- Kannada ASR ----------
print("Loading Kannada ASR model...")
kannada_processor = Wav2Vec2Processor.from_pretrained(ORIGINAL_MODEL_NAME)
kannada_model = Wav2Vec2ForCTC.from_pretrained(KANNADA_MODEL_DIR).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
kannada_model.eval()
print("Kannada ASR model loaded")

def record_with_vad(silence_duration=1.0, energy_threshold=500, timeout=10.0):
    """
    Records audio using device 1 (working microphone) with int16 dtype.
    Energy threshold is in int16 scale (max ~12000).
    """
    device = 1  # <-- CHANGED to device 1
    device_info = sd.query_devices(device, 'input')
    native_sr = int(device_info['default_samplerate'])
    audio_queue = queue.Queue()
    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())
    print("Recording... (speak now)")
    stream = sd.InputStream(device=device, samplerate=native_sr, channels=1, callback=callback, dtype='int16')
    stream.start()
    chunks = []
    last_voice = time.time()
    start_time = time.time()
    speech_started = False
    while True:
        if time.time() - start_time > timeout:
            break
        try:
            chunk = audio_queue.get(timeout=0.1)
            # chunk is int16
            energy = np.sqrt(np.mean(chunk.astype(np.float32)**2))
            if energy > energy_threshold:
                last_voice = time.time()
                if not speech_started:
                    print(f"Speech detected (energy={energy:.1f})")
                    speech_started = True
            else:
                if speech_started and (time.time() - last_voice) > silence_duration:
                    print("Silence detected, stopping.")
                    break
            chunks.append(chunk)
        except queue.Empty:
            continue
    stream.stop()
    stream.close()
    if not chunks:
        return None, native_sr
    audio_int16 = np.concatenate(chunks, axis=0).flatten()
    # Convert to float32 in range [-1, 1] for ASR
    audio = audio_int16.astype(np.float32) / 32768.0
    # Save debug recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(CAPTURES_DIR, f"last_audio_{timestamp}.wav")
    sf.write(wav_path, audio, native_sr)
    print(f"Saved recorded audio to {wav_path}")
    return audio, native_sr

def transcribe_kannada(audio, native_sr, target_sr=16000):
    if native_sr != target_sr:
        num_samples = int(len(audio) * target_sr / native_sr)
        audio = signal.resample(audio, num_samples).astype(np.float32)
    inputs = kannada_processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True).input_values.to(kannada_model.device)
    with torch.no_grad():
        logits = kannada_model(inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = kannada_processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
    return text

# ---------- Speech Reconstruction Thread (continuously loops until "ಸಾಕು") ----------
stop_speech = False
speech_exit_requested = False
speech_thread = None

def speech_loop():
    global stop_speech, speech_exit_requested
    while not stop_speech:
        time.sleep(0.5)
        _speak_text_internal("Please speak now.")
        time.sleep(0.3)
        audio, sr = record_with_vad(silence_duration=1.0, energy_threshold=500, timeout=10.0)
        if audio is None:
            print("No speech detected. Please try again.")
            continue
        text = transcribe_kannada(audio, sr)
        if not text or text.strip() == "":
            print("No text recognized. Please try again.")
            continue
        print(f"🗣️ KANNADA TEXT: {text}")
        if "ಸಾಕು" in text:
            print("User said 'ಸಾಕು'. Exiting speech mode.")
            speak_text("Exiting speech mode.")
            speech_exit_requested = True
            break
        _speak_kannada_internal(text)

# ---------- YOLO & OCR Engine ----------
print(f"Loading YOLO from {YOLO_MODEL_PATH}")
yolo = YOLO(YOLO_MODEL_PATH)
print("YOLO loaded")

print("Loading ONNX OCR engine...")
ocr_engine = RapidOCR(
    det_model_path=DET_MODEL_PATH,
    rec_model_path=REC_MODEL_PATH,
    rec_keys_path=DICT_PATH,
    det_db_thresh=0.3,
    det_db_box_thresh=0.3,
)
print("ONNX OCR engine ready")

def draw_boxes(frame, boxes, names):
    if boxes is None:
        return frame
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        label = f"{names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def get_book_box(boxes, names, frame_h, frame_w):
    for b in boxes:
        cls_name = names[int(b.cls[0])].lower()
        if cls_name in ("book", "document", "paper", "notebook", "magazine", "laptop", "tv"):
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            w = x2-x1; h = y2-y1
            if w>0 and h>0:
                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(frame_w, x2 + pad_x)
                y2 = min(frame_h, y2 + pad_y)
                return (x1,y1,x2,y2)
    return None

def get_largest_color_target(boxes, names):
    candidates = []
    for b in boxes:
        cls_name = names[int(b.cls[0])].lower()
        if cls_name in HUMAN_CLASSES or cls_name not in COLOR_TARGET_CLASSES: continue
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        candidates.append(((x2-x1)*(y2-y1), cls_name, (x1,y1,x2,y2)))
    if not candidates: return None,None,None
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2], candidates[0][0]

# ---------- Main ----------
def main():
    global frame_counter, last_boxes, last_names, last_annotated, tts_worker_thread
    global stop_speech, speech_exit_requested, speech_thread, cap

    tts_worker_thread = threading.Thread(target=_tts_worker, daemon=True)
    tts_worker_thread.start()

    mode = "idle"
    last_ocr_time = 0
    book_visible_since = None
    ocr_active = False
    ocr_display_start = 0
    occlusion_start = None
    memory_object_learned = False
    memory_target_visible_since = None
    memory_target_box = None
    memory_target_class = None
    last_color_speak = 0
    processing_document = False
    finding_cooldown_until = 0

    try:
        while True:
            # Ensure camera is open only when needed (not in speech mode)
            if mode != "speech":
                ensure_camera()
            else:
                if cap is not None:
                    release_camera()

            # Frame reading only if camera is available and not in speech mode
            if mode != "speech" and cap is not None:
                ret = False
                for attempt in range(3):
                    try:
                        ret, frame = cap.read()
                        if ret:
                            break
                    except Exception as e:
                        print(f"Camera read exception: {e}")
                    time.sleep(0.05)
                if not ret:
                    print("Camera read failed, reinitializing...")
                    init_camera()
                    continue
            else:
                frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
                annotated = frame.copy()
                ret = True

            if mode != "speech" and cap is not None:
                frame_counter += 1
                if frame_counter % YOLO_EVERY_N_FRAMES == 0 or last_boxes is None:
                    results = yolo(frame, verbose=False)[0]
                    last_boxes = results.boxes
                    last_names = results.names
                    annotated = results.plot()
                    last_annotated = annotated
                else:
                    annotated = draw_boxes(frame.copy(), last_boxes, last_names)
                boxes = last_boxes
                names = last_names
            else:
                boxes = []
                names = {}

            now = time.time()
            cv2.putText(annotated, f"MODE: {mode.upper()}", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(annotated, f"OCR: {'ACTIVE' if ocr_active else 'idle'}", (12,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0) if ocr_active else (0,200,255), 2)

            # Mode switching via occlusion
            if mode in ("idle","memory","finding","color","speech"):
                is_dark = False
                if mode != "speech" and cap is not None:
                    gray_mean = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    is_dark = gray_mean < DARK_MEAN_THRESHOLD
                if is_dark:
                    if occlusion_start is None:
                        occlusion_start = now
                    elif (now - occlusion_start) >= OCCLUSION_TRIGGER_SEC:
                        if mode == "idle":
                            mode = "memory"
                            memory_object_learned = False
                            memory_target_visible_since = None
                            memory_target_box = None
                            memory_target_class = None
                            speak_text("Entering memory mode.")
                            print("MEMORY MODE")
                        elif mode == "memory":
                            if not memory_object_learned:
                                speak_text("No object learned. Entering finding mode.")
                            else:
                                speak_text("Entering finding mode.")
                            mode = "finding"
                            finding_cooldown_until = 0
                        elif mode == "finding":
                            speak_text("Entering color detection mode.")
                            mode = "color"
                        elif mode == "color":
                            # Mode entry message – speak synchronously to avoid overlap
                            _speak_text_internal("Entering speech reconstruction mode.")
                            mode = "speech"
                            if speech_thread is None:
                                stop_speech = False
                                speech_exit_requested = False
                                speech_thread = threading.Thread(target=speech_loop, daemon=True)
                                speech_thread.start()
                                print("Speech reconstruction thread started")
                        elif mode == "speech":
                            if speech_thread is not None:
                                stop_speech = True
                                speech_thread.join(timeout=2.0)
                                speech_thread = None
                                print("Speech thread stopped")
                            speak_text("Returning to normal mode.")
                            mode = "idle"
                        occlusion_start = None
                else:
                    occlusion_start = None

            # Automatic exit from speech mode when speech_exit_requested (only on "ಸಾಕು")
            if mode == "speech" and speech_exit_requested:
                if speech_thread is not None:
                    stop_speech = True
                    speech_thread.join(timeout=2.0)
                    speech_thread = None
                    print("Speech thread stopped (exit requested)")
                speech_exit_requested = False
                stop_speech = False
                mode = "idle"
                cv2.imshow("MIRA Unified", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ---------- OCR, memory, finding, colour modes (unchanged) ----------
            if mode == "idle" and cap is not None:
                book_box = get_book_box(boxes, names, frame.shape[0], frame.shape[1])
                if book_box is not None:
                    x1,y1,x2,y2 = book_box
                    if (x2-x1) >= 50 and (y2-y1) >= 50:
                        if book_visible_since is None and not processing_document:
                            book_visible_since = now
                            processing_document = True
                            speak_text("Document detected")
                            optimize_camera_for_ocr()
                            refocus_camera()
                            time.sleep(0.2)
                        if processing_document:
                            stable = now - book_visible_since
                            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 3)
                            cv2.putText(annotated, f"Stable: {stable:.1f}s", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                            if stable >= MIN_STABLE_SECONDS and (now - last_ocr_time) > OCR_COOLDOWN:
                                speak_text("Processing text")
                                print(f"\nCapturing burst in {PRE_CAPTURE_DELAY}s...")
                                time.sleep(PRE_CAPTURE_DELAY)

                                burst, scores = [], []
                                prev_gray = None
                                for _ in range(BURST_COUNT):
                                    ok2, f2 = cap.read()
                                    if not ok2: continue
                                    crop = f2[y1:y2, x1:x2]
                                    if crop.size == 0: continue
                                    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                    motion = 0 if prev_gray is None else np.mean(cv2.absdiff(g, prev_gray))
                                    prev_gray = g
                                    if motion < 25:
                                        score = SHARPNESS_WEIGHT * sharpness(crop) + (1 - SHARPNESS_WEIGHT) * np.mean(g)
                                        burst.append(crop)
                                        scores.append(score)
                                    time.sleep(BURST_INTERVAL)

                                if burst:
                                    best_idx = int(np.argmax(scores))
                                    best_crop = burst[best_idx]
                                    proc = preprocess(best_crop)
                                    cv2.imshow("Raw Capture", best_crop)
                                    cv2.imshow("Preprocessed", proc)
                                    cv2.waitKey(1)

                                    result, _ = ocr_engine(proc)
                                    if result:
                                        lines = [item[1] for item in result if item[2] > 0.5]
                                        if lines:
                                            combined = " ".join(lines)
                                            print("\nEXTRACTED TEXT:")
                                            print(combined)
                                            save_result(best_crop, proc, lines)
                                            speak_text(combined)
                                        else:
                                            print("No confident text found")
                                            speak_text("No clear text detected")
                                    else:
                                        print("No text detected")
                                        speak_text("No text found")
                                else:
                                    print("No stable frames captured")
                                    speak_text("Hold steadier")

                                last_ocr_time = time.time()
                                ocr_active = True
                                ocr_display_start = now
                                book_visible_since = None
                                processing_document = False
                    else:
                        book_visible_since = None
                        processing_document = False
                else:
                    if processing_document:
                        print("Document disappeared – resetting detection")
                    book_visible_since = None
                    processing_document = False

            if ocr_active and (now - ocr_display_start > DISPLAY_OCR_RESULT_SEC):
                ocr_active = False

            if mode == "memory" and cap is not None:
                best_candidate = None
                best_area = 0
                best_box = None
                best_class = None
                for b in boxes:
                    cls_name = names[int(b.cls[0])].lower()
                    if cls_name in HUMAN_CLASSES: continue
                    if float(b.conf[0]) < 0.3: continue
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    area = (x2-x1)*(y2-y1)
                    if area > best_area:
                        best_area = area
                        best_box = (x1,y1,x2,y2)
                        best_class = cls_name
                        best_candidate = frame[y1:y2, x1:x2]
                if best_candidate is not None:
                    if memory_target_box is not None:
                        iou_val = iou(best_box, memory_target_box)
                        if iou_val > 0.5 and best_class == memory_target_class:
                            if memory_target_visible_since is None:
                                memory_target_visible_since = now
                            else:
                                stable_time = now - memory_target_visible_since
                                cv2.putText(annotated, f"Target stable: {stable_time:.1f}s", (best_box[0], best_box[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                                if stable_time >= 1:
                                    print(f"Object {best_class} stable")
                                    emb = extract_embedding(best_candidate)
                                    name = listen_once("Please say the name of this item now.", timeout=5)
                                    if name:
                                        confirm = listen_once(f"Did you say {name}? Say yes or no.", timeout=5)
                                        if "yes" in confirm.lower():
                                            save_memory(name, emb)
                                            memory_object_learned = True
                                            speak_text("Cover the camera for three seconds to enter finding mode.")
                                    memory_target_visible_since = None
                                    memory_target_box = None
                                    memory_target_class = None
                        else:
                            memory_target_visible_since = now
                            memory_target_box = best_box
                            memory_target_class = best_class
                    else:
                        memory_target_visible_since = now
                        memory_target_box = best_box
                        memory_target_class = best_class
                else:
                    memory_target_visible_since = None
                    memory_target_box = None
                    memory_target_class = None

            if mode == "finding" and cap is not None:
                now_time = time.time()
                if now_time >= finding_cooldown_until:
                    memories = load_memories()
                    if memories:
                        best_match = None
                        best_sim = 0.0
                        for b in boxes:
                            cls_name = names[int(b.cls[0])].lower()
                            if cls_name in HUMAN_CLASSES: continue
                            x1,y1,x2,y2 = map(int, b.xyxy[0])
                            crop = frame[y1:y2, x1:x2]
                            if crop.size == 0: continue
                            emb = extract_embedding(crop)
                            # Locate your "finding" mode matching block inside main():
                            for mem in memories:
                                # 1. Extract the string label name out of the dictionary key
                                target_name = mem["name"]
                                
                                # 2. Extract the raw embedding list vector matching the database column signature
                                db_embedding = mem["embedding"]
                                
                                # 3. Feed the unpacked database embedding into your Cosine Similarity math engine
                                sim = cos_sim(emb, db_embedding)
                                
                                if sim > MEM_SIM_THRESHOLD and sim > best_sim:
                                    best_sim = sim
                                    best_match = target_name
                        if best_match:
                            speak_text(f"{best_match} is in the line of sight.")
                            print(f"MATCH FOUND: {best_match} (sim={best_sim:.2f})")
                            finding_cooldown_until = now_time + 5.0

            if mode == "color" and cap is not None:
                obj_name, box, _ = get_largest_color_target(boxes, names)
                if obj_name and box and (now - last_color_speak) > COLOR_SPEAK_COOLDOWN:
                    x1,y1,x2,y2 = box
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        color, conf = get_dominant_color_simple(crop)
                        if conf >= COLOR_MIN_CONFIDENCE:
                            cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,0,255), 4)
                            cv2.putText(annotated, f"CLOSEST: {color.upper()} {obj_name} ({conf:.0%})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
                            print(f"COLOR: {obj_name} → {color} (conf: {conf:.2%})")
                            speak_text(f"Your object is {color}")
                            last_color_speak = now

            cv2.imshow("MIRA Unified", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if speech_thread is not None:
            stop_speech = True
            speech_thread.join(timeout=2.0)
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        tts_queue.put((None, (), {}))
        if tts_worker_thread:
            tts_worker_thread.join(timeout=2)
        print("MIRA shutdown")

if __name__ == "__main__":
    main()
