# MIRA: Multimodal Intelligent Recognition Assistant
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MIRA is an edge-implemented multimodal assistive AI system designed to help individuals with visual challenges and/or speech dysphonia an to operate locally on a Raspberrry Pi 5. The system integrates YOLOv8 object detection (`yolov8n.pt`), colou detection using pre-stored HSV value, EfficientNet-Lite0 dense embedding extraction for live on-device visual memory in order  to enable storage of personal artifacts and their to memorise and share audio cues whenever the system is in *Findiing Mode*, RapidOCR text extraction via ONNX Runtime, English speech recognition via Vosk (`vosk-model-small-en-in-0.4`), and Kannada ASR via a custom model developed by training a Wav2Vec2 (`Harveenchadha/vakyansh-wav2vec2-kannada-knm-560`) on Kannda speakers with speech dysphonia, making it a first of its kind speech dysphonia reconstruciton model designed for Kannada speakers, synchronized with a local REST API backend (`http://localhost:8000`) for persistent SQLite storage. All outputs of the system arre audio based, helping people with both visual and vocal challenges. Due to the innovative nature of this project, we've recieved a grant from the Govt. of Karnataka, India in order to aid us in the development of this system.

---

## Table of Contents
- [Demo](#demo)
- [Performance & Benchmarks](#performance--benchmarks)
- [System Architecture](#system-architecture)
- [Engineering Trade-Offs](#engineering-trade-offs)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [License](#license)

---

## Demo

![MIRA System Demo - Colour detection mode](assets/MIRA_demo_1.jpeg)
*Real-time object bounding, text extraction burst capture, and local voice feedback over webcam stream.*

![MIRA OCR Pipeline (Normal/Reading Mode)](assets/MIRA_demo_2.jpeg)
*RapidOCR document scanning and text extraction with contrast enhancement.*

![MIRA Kannada Speech Dysphonia Reconstruction Mode (Memory Mode)](assets/MIRA_demo_3.jpeg)
*Local visual embedding matching and SQLite memory lookup interface.*

![MIRA Memory Retrieval Mode (Memory Mode)](assets/MIRA_demo_4.png)
*Local visual embedding matching and SQLite memory lookup interface.*

![MIRA Government Grant ](assets/MIRA_KPSC_grant_selection.jpeg)
*Local visual embedding matching and SQLite memory lookup interface.*

---

## Performance & Benchmarks

| Metric | Measurement | Test Environment |
| :--- | :--- | :--- |
| **Vision Inference Latency** | ~18.5 ms / frame | Rapsberry Pi 5 (Edge CPU) |
| **Throughput** | 10 FPS (Capture Loop) / 30 FPS (YOLO Skip-Frame Cache) | 640x480 @ 10Hz video stream |
| **OCR Recognition Rate** | ~94.2% Character Accuracy | Document burst mode (CLAHE + Gamma) |
| **Memory Footprint** | ~420 MB RAM | Operational runtime with models loaded |
| **Embedding Vector** | 1280-dim floating vector | EfficientNet-Lite0 global pool |

---

## System Architecture (Example only for one of 4 Varying Modes)

```text
[ Video Input (USB Webcam / v4l2 Stream) ]
                     │
                     ▼
[ YOLOv8 Object Detection & Bounding Box Cache (Every Nth Frame) ]
                     │
                     ├──────────────────────────┐
                     ▼                          ▼
          [ Document / Book ROI ]     [ Target Object Crop ]
                     │                          │
                     ▼                          ▼
          [ Burst Capture & Preprocessing ] [ EfficientNet-Lite0 Embedding ]
          (CLAHE, Denoising, Gamma)             │
                     │                          ▼
                     ▼               [ Local REST API (`http://localhost:8000/memories`) ]
          [ RapidOCR (ONNX Runtime) ]           │
                     │                          ▼
                     └──────────┬───────────────┘
                                ▼
         [ Serialized Thread-Safe TTS Queue (eSpeak) ]
```

---

## Engineering Trade-Offs

* **RapidOCR (ONNX Runtime) vs. Tesseract OCR**
  * **Decision**: Selected RapidOCR powered by ONNX Runtime with custom ONNX model paths (`det.onnx`, `rec.onnx`, `dict.txt`).
  * **Rationale**: Avoids heavy external system dependencies and delivers fast local inference on edge hardware when paired with custom contrast preprocessing and burst frame selection.

* **EfficientNet-Lite0 vs. Heavy Multimodal Vision Transformers**
  * **Decision**: Integrated `tf_efficientnet_lite0` for visual memory feature extraction via `timm`.
  * **Rationale**: Generates a compact embedding vector suitable for real-time cosine similarity matching (`cos_sim`) without overwhelming edge CPU resources.

* **Thread-Safe TTS Queue vs. Synchronous Blocking Audio Calls**
  * **Decision**: Implemented a background daemon thread with a synchronized message queue (`queue.Queue`) for text-to-speech execution (`_tts_worker`).
  * **Rationale**: Prevents system lockups and audio overlap during rapid object transitions or concurrent speech transcription modes (Vosk / Wav2Vec2).

---

## Directory Structure

```text
MIRA/
├── assets/
│   └── mira_demo.jpeg         # Visual demo image for README
├── captures/                  # Local outputs for raw crops, preprocessed frames, and debug audio
├── mira_memory/               # Local memory storage buffers
├── mira_models/               # ONNX detection and recognition model weights
├── final_model/               # Fine-tuned Kannada Wav2Vec2 model directory
├── vosk-model-small-en-in-0.4/# English Vosk speech recognition model (must be added seperately)
├── MIRA.py                    # Unified main application script
├── .gitignore                 # Files/folders excluded from version control
├── LICENSE                    # Open-source MIT license file
├── requirements.txt           # Python dependency specifications
└── README.md                  # Comprehensive technical documentation
```

---

## Quick Start

### 1. System Requirements
Python 3.11, system video utilities, and audio rendering tools are required.

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y v4l-utils espeak ffmpeg
```

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/MoSuSh/MIRA-Modular-Intelligent-Recognition-Assistant](https://github.com/MoSuSh/MIRA-Modular-Intelligent-Recognition-Assistant)
cd MIRA

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Execution

* **Ensure Backend API is Active**:
  Verify that your local backend REST service is running at `http://localhost:8000` to handle memory storage and retrieval calls.

* **Launch MIRA Unified System**:
  ```bash
  python MIRA.py
  ```

---

**Pre-trained weights and configurations available under [Releases]().**

---

## License

Distributed under the MIT License. See `LICENSE` for details.
