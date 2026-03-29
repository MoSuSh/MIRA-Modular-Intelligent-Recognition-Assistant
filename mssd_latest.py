# ============================================================
# MIRA UNIFIED (SSD version) – with FINDING MODE DEBUG
#                + memory stabilization (1.5s hold)
#                + SIMPLE HSV COLOR DETECTION
# ============================================================

import os
import cv2
import time
import numpy as np
import re
from datetime import datetime
from paddleocr import PaddleOCR
import torch
import timm
import json
import queue
import sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer

# -----------------------------
# GLOBAL SETTINGS
# -----------------------------
FRAME_W, FRAME_H = 1280, 720
MIN_STABLE_SECONDS = 1.5
PRE_CAPTURE_DELAY = 0.8
BURST_COUNT = 6
BURST_INTERVAL = 0.07
UPSCALE_FACTOR = 1.1
OCR_COOLDOWN = 3.0
DISPLAY_OCR_RESULT_SEC = 6
SHARPNESS_WEIGHT = 0.7
CAPTURES_DIR = "captures"
os.makedirs(CAPTURES_DIR, exist_ok=True)

MEMORY_DIR = "mira_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

OCCLUSION_TRIGGER_SEC = 3.0
DARK_MEAN_THRESHOLD = 25
MEM_SIM_THRESHOLD = 0.60   # lowered even more for easier matching

HUMAN_CLASSES = {"person","man","woman","boy","girl","face","human","head","body"}
COLOR_TARGET_CLASSES = {
    # Your existing items...
    "bottle", "cup", "bowl", "plate", "knife", "fork", "spoon",
    "shirt", "pants", "backpack", "handbag", "umbrella",
    "laptop", "cell phone", "remote", "keyboard", "mouse",
    "book", "banana", "apple", "orange", "carrot",
    
    # NEW additions from COCO
    "wine glass", "sandwich", "hot dog", "pizza", "donut", "cake",
    "broccoli", "tv", "microwave", "oven", "toaster", "sink", "refrigerator",
    "vase", "clock", "scissors", "teddy bear", "hair drier", "toothbrush",
    "frisbee", "kite", "sports ball", "skateboard", "surfboard",
    "baseball bat", "baseball glove", "tennis racket",
    "tie", "suitcase"
}
COLOR_SPEAK_COOLDOWN = 3.0
COLOR_MIN_CONFIDENCE = 0.15   # minimum fraction of pixels that must match a colour

TTS_ENABLED = True
TTS_RATE = 150
TTS_VOLUME = 1.0
MIN_SPEAK_CONFIDENCE = 0.1

VOSK_MODEL_PATH = r"C:\Users\moham\Desktop\MIRA\vosk-model-small-en-in-0.4"

# ============================================================
# TTS FUNCTIONS (unchanged)
# ============================================================
def init_tts():
    if not TTS_ENABLED: return None
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        print(f"🔊 TTS ready ({TTS_RATE} wpm)")
        engine.stop()
        return None
    except Exception as e:
        print(f"❌ TTS init failed: {e}")
        return None

def speak_text(text):
    if not TTS_ENABLED: return
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"❌ Speak error: {e}")

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?;])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 1]

def speak_ocr_text(long_text):
    print(f"🔊 OCR text ({len(long_text)} chars)")
    sentences = split_into_sentences(long_text)
    if not sentences:
        print("⚠️ No sentences found")
        return
    for i, sentence in enumerate(sentences):
        print(f"🔊 Sentence {i+1}: {sentence}")
        speak_text(sentence)
        time.sleep(0.25)
    print("✅ All sentences spoken")

def shutdown_tts():
    print("✅ TTS shutdown")

# ============================================================
# OCR PIPELINE (unchanged)
# ============================================================
def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = np.mean(gray)
    gamma = 1.8 if m < 60 else 1.4 if m < 100 else 0.95 if m > 200 else 1.0
    inv = 1.0 / gamma
    table = np.array([(i/255)**inv * 255 for i in range(256)], np.uint8)
    gray = cv2.LUT(gray, table)
    clahe = cv2.createCLAHE(1.1, (8,8))
    gray = clahe.apply(gray)
    den = cv2.fastNlMeansDenoising(gray, None, 3, 7, 21)
    blur = cv2.GaussianBlur(den, (3,3), 0)
    sharp = cv2.addWeighted(den, 1.05, blur, -0.05, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    clean = cv2.morphologyEx(sharp, cv2.MORPH_OPEN, kernel)
    if UPSCALE_FACTOR != 1:
        clean = cv2.resize(clean, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)

def clean_text(t):
    t = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', t)
    t = re.sub(r'[^A-Za-z0-9.,;:!?\'"()\-\s]', ' ', t)
    return re.sub(r'\s{2,}', ' ', t).strip()

def save_result(raw, proc, lines):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rp = os.path.join(CAPTURES_DIR, f"cap_{ts}.jpg")
    pp = os.path.join(CAPTURES_DIR, f"cap_{ts}_proc.jpg")
    tp = os.path.join(CAPTURES_DIR, f"cap_{ts}.txt")
    cv2.imwrite(rp, raw)
    cv2.imwrite(pp, proc)
    with open(tp, "w", encoding="utf-8") as f:
        for L in lines:
            f.write(L + "\n")

# ============================================================
# EMBEDDINGS (EfficientNet-lite0)
# ============================================================
print("🧠 Loading embedding backbone...")
embed_model = timm.create_model("tf_efficientnet_lite0", pretrained=True, num_classes=0, global_pool="avg")
embed_model.eval()

def extract_embedding(img):
    img_resized = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        emb = embed_model(img_tensor).numpy().flatten()
    return emb.tolist()

def save_memory(name, embedding):
    mem_path = os.path.join(MEMORY_DIR, f"{name}.json")
    with open(mem_path, "w") as f:
        json.dump({"name": name, "embedding": embedding}, f)
    print(f"💾 Saved memory: {mem_path}")

def load_memories():
    memories = []
    for file in os.listdir(MEMORY_DIR):
        if file.endswith(".json"):
            with open(os.path.join(MEMORY_DIR, file), "r") as f:
                memories.append(json.load(f))
    return memories

def cos_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0: return 0.0
    return float(np.dot(a, b) / denom)

# ============================================================
# Helper function for IoU (used in memory stabilisation)
# ============================================================
def iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# ============================================================
# SIMPLE HSV COLOUR DETECTION (no k-means, no Gray-World)
# ============================================================

# Define HSV ranges for basic colours (you may need to tune these)
COLOR_RANGES = {
    "red":    [([0,50,50], [10,255,255]), ([170,50,50], [180,255,255])],
    "green":  [([40,40,40], [80,255,255])],
    "blue":   [([100,50,50], [130,255,255])],
    "yellow": [([20,50,50], [35,255,255])],
    "orange": [([10,50,50], [20,255,255])],
    "purple": [([130,50,50], [160,255,255])],
    "pink":   [([160,50,50], [170,255,255])],
    "brown":  [([10,50,20], [20,255,150])],
    "gray":   [([0,0,40], [180,40,220])],
    "black":  [([0,0,0], [180,255,50])],
    "white":  [([0,0,220], [180,40,255])]
}

def get_dominant_color_simple(crop):
    """
    Determine the dominant colour of a crop by counting pixels
    that fall into each HSV range.
    Returns (color_name, confidence) where confidence = fraction of pixels in best range.
    """
    if crop.size == 0:
        return "none", 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_pixels = crop.shape[0] * crop.shape[1]

    best_color = "unknown"
    best_count = 0

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask += cv2.inRange(hsv, lower, upper)
        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count = count
            best_color = color_name

    confidence = best_count / total_pixels
    return best_color, confidence

# ============================================================
# Helper function to get largest target object for COLOR mode
# ============================================================
def get_largest_color_target(boxes, names, frame):
    candidates = []
    for b in boxes:
        cls_name = names[int(b.cls[0])].lower()
        if cls_name in HUMAN_CLASSES: continue
        if cls_name in COLOR_TARGET_CLASSES:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            area = (x2-x1)*(y2-y1)
            candidates.append((area, cls_name, (x1,y1,x2,y2)))
    if not candidates: return None,None,None
    candidates.sort(reverse=True)
    _, obj_name, box = candidates[0]
    return obj_name, box, _

# ============================================================
# VOSK VOICE
# ============================================================
if os.path.exists(VOSK_MODEL_PATH):
    print("🗣️ Loading Vosk model...")
    vosk_model = VoskModel(VOSK_MODEL_PATH)
else:
    print("❌ Vosk model not found")
    vosk_model = None

def listen_once(prompt=None, timeout=8.0):
    if vosk_model is None: return ""
    if prompt: speak_text(prompt)
    q = queue.Queue()
    def callback(indata, frames, time_info, status):
        if status: print(status)
        q.put(bytes(indata))
    rec = KaldiRecognizer(vosk_model, 16000)
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype="int16", channels=1, callback=callback):
            print("🎤 Listening...")
            start = time.time()
            final_text = ""
            while time.time()-start < timeout:
                try:
                    data = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    final_text = res.get("text","").strip()
                    break
            if not final_text:
                res = json.loads(rec.FinalResult())
                final_text = res.get("text","").strip()
    except Exception as e:
        print(f"🎤 Vosk error: {e}")
        final_text = ""
    print(f"🗣️ USER SAID: {final_text}")
    return final_text

# ============================================================
# SSD MODEL INIT
# ============================================================
print("🚀 Loading MobileNetV3-SSD model...")
SSD_MODEL_PATH = "C:\Users\moham\Desktop\MIRA\ssd_mobilenet_v3_large_coco_2020_01_14\frozen_inference_graph.pb"
SSD_CONFIG_PATH = "C:\Users\moham\Desktop\MIRA\ssd_mobilenet_v3_large_coco_2020_01_14\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
if not os.path.exists(SSD_MODEL_PATH) or not os.path.exists(SSD_CONFIG_PATH):
    print("❌ SSD model files missing.")
    exit(1)

net = cv2.dnn.readNetFromTensorflow(SSD_MODEL_PATH, SSD_CONFIG_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
           "fire hydrant","street sign","stop sign","parking meter","bench","bird","cat","dog","horse","sheep",
           "cow","elephant","bear","zebra","giraffe","hat","backpack","umbrella","shoe","eye glasses",
           "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
           "skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup","fork","knife","spoon",
           "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
           "cake","chair","couch","potted plant","bed","mirror","dining table","window","desk","toilet",
           "door","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster",
           "sink","refrigerator","blender","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

class Detection:
    def __init__(self, cls_id, xyxy, conf):
        self.cls = [cls_id]
        self.xyxy = [xyxy]
        self.conf = conf

print("✅ SSD model loaded")

# ============================================================
# PADDLEOCR & CAMERA
# ============================================================
ocr = PaddleOCR(use_angle_cls=True, lang="en")
init_tts()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS, 30)
print("📷 Camera ready. Q to quit.")

# ============================================================
# STATE VARIABLES
# ============================================================
mode = "idle"
last_ocr_time = 0
book_visible_since = None
ocr_active = False
ocr_display_start = 0
occlusion_start = None
memory_object_learned = False
last_color_speak = 0

# memory stabilisation variables
memory_target_visible_since = None
memory_target_box = None
memory_target_class = None

# ============================================================
# DRAW FUNCTION
# ============================================================
def draw_detections(frame, detections, class_names):
    colors = np.random.uniform(0,255,size=(len(class_names),3))
    for d in detections:
        cls_id = d.cls[0]
        conf = d.conf
        x1,y1,x2,y2 = d.xyxy[0]
        label = f"{class_names[cls_id]}: {conf:.2f}"
        color = colors[cls_id % len(colors)]
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        (w,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-h-5), (x1+w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return frame

# ============================================================
# MAIN LOOP
# ============================================================
try:
    while True:
        ok, frame = cap.read()
        if not ok: break

        # SSD inference
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        boxes = []
        h,w = frame.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence > 0.5:
                class_id = int(detections[0,0,i,1])
                if class_id == 0: continue
                class_id -= 1
                if class_id >= len(CLASSES): continue
                x1 = int(detections[0,0,i,3] * w)
                y1 = int(detections[0,0,i,4] * h)
                x2 = int(detections[0,0,i,5] * w)
                y2 = int(detections[0,0,i,6] * h)
                x1,y1 = max(0,x1), max(0,y1)
                x2,y2 = min(w-1,x2), min(h-1,y2)
                if x2>x1 and y2>y1:
                    boxes.append(Detection(class_id, [x1,y1,x2,y2], confidence))

        names = {i: CLASSES[i] for i in range(len(CLASSES))}
        annotated = draw_detections(frame.copy(), boxes, CLASSES)

        now = time.time()
        cv2.putText(annotated, f"MODE: {mode.upper()}", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        ocr_status = "OCR: ACTIVE" if ocr_active else "OCR: idle"
        ocr_color = (0,200,0) if ocr_active else (0,200,255)
        cv2.putText(annotated, ocr_status, (12,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ocr_color, 2)

        gray_mean = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        is_dark = gray_mean < DARK_MEAN_THRESHOLD

        # Mode switching
        if mode in ("idle","memory","finding","color"):
            if is_dark:
                if occlusion_start is None:
                    occlusion_start = now
                elif now - occlusion_start >= OCCLUSION_TRIGGER_SEC:
                    if mode == "idle":
                        mode = "memory"
                        memory_object_learned = False
                        # Reset memory stabilisation when entering memory mode
                        memory_target_visible_since = None
                        memory_target_box = None
                        memory_target_class = None
                        speak_text("Entering memory mode.")
                    elif mode == "memory":
                        if not memory_object_learned:
                            speak_text("No object learned. Entering finding mode.")
                        else:
                            speak_text("Entering finding mode.")
                        mode = "finding"
                    elif mode == "finding":
                        speak_text("Entering color detection mode.")
                        mode = "color"
                    elif mode == "color":
                        speak_text("Returning to normal mode.")
                        mode = "idle"
                    occlusion_start = None
            else:
                occlusion_start = None

        # Book detection for OCR
        book_box = None
        for b in boxes:
            cls_name = names[int(b.cls[0])].lower()
            if cls_name in ("book","document","paper","notebook","magazine"):
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                pad = 20
                book_box = (max(0,x1-pad), max(0,y1-pad), min(frame.shape[1],x2+pad), min(frame.shape[0],y2+pad))
                break

        # IDLE -> OCR
        if mode == "idle":
            if book_box:
                if book_visible_since is None:
                    book_visible_since = now
                    speak_text("Document detected")
                x1,y1,x2,y2 = book_box
                stable = now - book_visible_since
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 3)
                cv2.putText(annotated, f"Stable: {stable:.1f}s", (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                if stable >= MIN_STABLE_SECONDS and (now - last_ocr_time) > OCR_COOLDOWN:
                    speak_text("Processing text")
                    time.sleep(PRE_CAPTURE_DELAY)
                    burst, scores, prev_gray = [], [], None
                    for _ in range(BURST_COUNT):
                        ok2, f2 = cap.read()
                        if not ok2: continue
                        crop = f2[y1:y2, x1:x2]
                        if crop.size == 0: continue
                        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        motion = 0 if prev_gray is None else np.mean(cv2.absdiff(g, prev_gray))
                        prev_gray = g
                        if motion < 25:
                            score = SHARPNESS_WEIGHT * sharpness(crop) + (1-SHARPNESS_WEIGHT)*np.mean(g)
                            burst.append(crop)
                            scores.append(score)
                        time.sleep(BURST_INTERVAL)
                    if burst:
                        best = burst[np.argmax(scores)]
                        proc = preprocess(best)
                        cv2.imshow("Raw", best)
                        cv2.imshow("Processed", proc)
                        cv2.waitKey(1)
                        try:
                            ocr_res = ocr.predict(proc)
                        except Exception as e:
                            print(f"OCR error: {e}")
                            speak_text("OCR failed")
                            book_visible_since = None
                            continue
                        lines = []
                        if isinstance(ocr_res, list):
                            for item in ocr_res:
                                if isinstance(item, dict):
                                    for t,c in zip(item.get("rec_texts",[]), item.get("rec_scores",[])):
                                        lines.append((t,float(c)))
                                elif isinstance(item, list):
                                    for region in item:
                                        if isinstance(region,list) and len(region)>=2:
                                            try:
                                                t,c = region[1]
                                                lines.append((t,float(c)))
                                            except: pass
                        speak_lines, save_lines = [], []
                        for text_raw, conf in lines:
                            cleaned = clean_text(text_raw)
                            if cleaned and conf >= MIN_SPEAK_CONFIDENCE:
                                speak_lines.append(cleaned)
                                save_lines.append(f"{cleaned} (conf: {conf:.2f})")
                        if speak_lines:
                            for ln in speak_lines: print(f"  {ln}")
                            save_result(best, proc, save_lines)
                            combined = " ".join(speak_lines)
                            speak_ocr_text(combined)
                        else:
                            speak_text("No clear text detected")
                        last_ocr_time = now
                        ocr_active = True
                        ocr_display_start = now
                    else:
                        speak_text("Hold steadier")
                    book_visible_since = None
            else:
                book_visible_since = None

        if ocr_active and (now - ocr_display_start > DISPLAY_OCR_RESULT_SEC):
            ocr_active = False

        # ==================== MEMORY MODE (with stabilization) ====================
        if mode == "memory":
            # Find the best non-human object (largest area, confidence > 0.3)
            best_candidate = None
            best_area = 0
            best_box = None
            best_class = None
            threshold = 0.3

            for b in boxes:
                cls_name = names[int(b.cls[0])].lower()
                if cls_name in HUMAN_CLASSES:
                    continue
                if b.conf < threshold:
                    continue
                x1_t, y1_t, x2_t, y2_t = map(int, b.xyxy[0])
                area = (x2_t - x1_t) * (y2_t - y1_t)
                if area > best_area:
                    best_area = area
                    best_box = (x1_t, y1_t, x2_t, y2_t)
                    best_class = cls_name
                    best_candidate = frame[y1_t:y2_t, x1_t:x2_t]

            # If we have a candidate, check if it's the same as the previous target
            if best_candidate is not None:
                # If we have a previous target, check if the new box overlaps enough
                if memory_target_box is not None:
                    iou_val = iou(best_box, memory_target_box)
                    if iou_val > 0.5 and best_class == memory_target_class:
                        # Same object – continue timer
                        if memory_target_visible_since is None:
                            memory_target_visible_since = now
                        else:
                            stable_time = now - memory_target_visible_since
                            # Display stabilisation progress on frame
                            cv2.putText(annotated, f"Target stable: {stable_time:.1f}s", 
                                        (best_box[0], best_box[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                            if stable_time >= 1.5:
                                # Stabilised – proceed to learn
                                print(f"✅ Object {best_class} stable for 1.5s. Starting capture...")
                                # Capture burst and ask for name
                                burst = []
                                for _ in range(BURST_COUNT):
                                    ok2, f2 = cap.read()
                                    if ok2:
                                        burst.append(f2[best_box[1]:best_box[3], best_box[0]:best_box[2]])
                                    time.sleep(BURST_INTERVAL)
                                best_capture = burst[-1] if burst else best_candidate
                                emb = extract_embedding(best_capture)
                                print("🧠 EMBEDDING EXTRACTED")

                                name = listen_once("Please say the name of this item now.").strip()
                                if not name:
                                    speak_text("I did not catch the name. Try again.")
                                else:
                                    speak_text(f"Did you say {name}? Say yes or no.")
                                    confirm = listen_once().lower()
                                    if "yes" in confirm:
                                        speak_text(f"Saving {name}.")
                                        save_memory(name, emb)
                                        memory_object_learned = True
                                        speak_text("Cover the camera for three seconds to enter finding mode.")
                                    else:
                                        speak_text("Name not confirmed. Try again.")
                                # After learning, exit memory mode (optional)
                                # Reset target
                                memory_target_visible_since = None
                                memory_target_box = None
                                memory_target_class = None
                    else:
                        # New object appeared – reset timer and target
                        memory_target_visible_since = now
                        memory_target_box = best_box
                        memory_target_class = best_class
                else:
                    # First object in this memory session
                    memory_target_visible_since = now
                    memory_target_box = best_box
                    memory_target_class = best_class
            else:
                # No candidate – reset timer and target
                memory_target_visible_since = None
                memory_target_box = None
                memory_target_class = None

        # FINDING MODE – now with debug prints and lower threshold
        if mode == "finding":
            memories = load_memories()
            print(f"🔍 Loaded {len(memories)} memories.")  # DEBUG
            if memories:
                for b in boxes:
                    cls_name = names[int(b.cls[0])].lower()
                    if cls_name in HUMAN_CLASSES:
                        continue
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_emb = extract_embedding(crop)
                    print(f"Checking object: {cls_name}")  # DEBUG
                    for mem in memories:
                        sim = cos_sim(crop_emb, mem["embedding"])
                        print(f"  vs {mem['name']}: sim = {sim:.3f}")  # DEBUG
                        if sim > MEM_SIM_THRESHOLD:
                            speak_text(f"{mem['name']} is in the line of sight.")
                            print(f"🔔 MATCH FOUND: {mem['name']} (sim={sim:.3f})")
                            time.sleep(2)
                            break

        # ==================== COLOR MODE – SIMPLE HSV RANGES ====================
        if mode == "color":
            obj_name, box, area = get_largest_color_target(boxes, names, frame)
            if obj_name and box and (now - last_color_speak) > COLOR_SPEAK_COOLDOWN:
                x1,y1,x2,y2 = box
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    color_name, conf = get_dominant_color_simple(crop)
                    if conf >= COLOR_MIN_CONFIDENCE:
                        cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,0,255), 4)
                        cv2.putText(annotated, f"CLOSEST: {color_name.upper()} {obj_name} ({conf:.0%})",
                                    (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
                        speak_text(f"Your {obj_name} is {color_name}")
                        print(f"🎨 COLOR: {obj_name} → {color_name} ({conf:.2%})")
                        last_color_speak = now

        cv2.imshow("MIRA Unified", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")
finally:
    shutdown_tts()
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Shutdown complete")