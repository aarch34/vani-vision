# 🎓 Vāṇī-Vision

> **Empowering Every Learner with an Offline AI Socratic Tutor in Their Own Language**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Offline](https://img.shields.io/badge/Runs-100%25%20Offline-green?style=flat-square)](.)
[![CPU Only](https://img.shields.io/badge/CPU-Only%20%7C%20No%20GPU-orange?style=flat-square)](.)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

---

## 🌟 What is Vāṇī-Vision?

**Vāṇī-Vision** is an offline-first, AI-powered educational tutoring system that transforms physical textbooks and handwritten problems into interactive, multilingual Socratic learning experiences — running entirely on a basic laptop with no internet connection.

Designed for **rural and government-school students across Bharat**, it bridges the gap between English-medium learning materials and students who think in Hindi, Kannada, Tamil, or Telugu.

---

## 🚨 The Problem We Solve

| Problem | Impact |
|---|---|
| Learning materials in English, students think regionally | Comprehension drops by 40–60% |
| No quality tutors in rural areas | 65% of students lack personalized guidance |
| AI tools require internet + GPU | Inaccessible to 500M+ offline Indians |
| Students memorize, not understand | Zero critical thinking skills |

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📷 **Webcam OCR** | Capture textbook pages or handwritten problems via webcam |
| 🧠 **Socratic AI** | Never gives answers — guides with targeted questions |
| 🌐 **Multilingual** | Hindi, Kannada, Tamil, Telugu, English |
| 📊 **Comprehension Meter** | Real-time understanding score (0–100%) |
| ⚡ **2–4 sec Response** | Optimized for CPU inference |
| 🔒 **100% Offline** | No internet, no cloud, no data leakage |
| 🖥️ **Runs on i3 + 8GB RAM** | No GPU required |

---

## 🏗️ System Architecture

```
📷 Webcam / Image Upload
        │
        ▼
┌─────────────────────┐
│   OCR Engine        │  ← PaddleOCR (CPU)
│   (ocr_engine.py)   │     Adaptive threshold preprocessing
│                     │     Bounding box annotation
└────────┬────────────┘
         │  Extracted Text
         ▼
┌─────────────────────┐
│  Subject Detector   │  ← Keyword matching
│ (socratic_prompt.py)│     Detects: Math / Physics / Chem / Bio
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Socratic Prompt    │  ← Dynamic system prompt
│   Generator         │     Mode: Scaffolded → Socratic → Deep → Hint
│ (socratic_prompt.py)│     Based on comprehension score
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Local LLM          │  ← Phi-3 Mini Q4 GGUF
│  (llm_engine.py)    │     llama-cpp-python, CPU only
│                     │     ~2–4 sec response time
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Multilingual Engine │  ← curated phrase dict + deep-translator
│ (multilingual.py)   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Understanding Meter│  ← Heuristic + optional LLM scoring
│(understanding_meter)│     Drives hint escalation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Streamlit UI      │  ← Dark glassmorphism, responsive
│   (app.py)          │     Camera panel + Chat panel + Meter
└─────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| **Frontend** | Streamlit 1.32 | Rapid Python-native UI |
| **OCR** | PaddleOCR 2.7 | Best handwriting accuracy, CPU-friendly |
| **LLM** | Phi-3 Mini 4K (Q4 GGUF) | 2.3GB, 2–4s on i3, multilingual |
| **Inference** | llama-cpp-python | Pure CPU GGUF runtime |
| **Translation** | deep-translator | Lightweight, cacheable |
| **Vision** | OpenCV 4.9 | Image preprocessing |
| **Language** | Python 3.10+ | Cross-platform |

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/vani-vision.git
cd vani-vision
pip install -r requirements.txt
```

### 2. Download the Model

Download **Phi-3 Mini Q4 GGUF** (~2.3 GB) from HuggingFace:

```
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
```

Place the file at:
```
vani-vision/models/phi-3-mini-4k-instruct-q4.gguf
```

### 3. Run

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
vani-vision/
├── app.py                  # Streamlit main UI
├── config.py               # All configuration constants
├── ocr_engine.py           # Webcam capture + PaddleOCR
├── llm_engine.py           # Phi-3 Mini inference
├── socratic_prompt.py      # Prompt engineering + subject detection
├── multilingual.py         # Language phrases + translation
├── understanding_meter.py  # Comprehension scoring
├── requirements.txt        # Python dependencies
├── models/                 # (Place GGUF model here)
└── README.md
```

---

## 💡 AI Behavior: Socratic Method Rules

The AI **never** gives direct answers. It always:

1. Acknowledges what the student said
2. Asks exactly **one** guiding question
3. Adapts difficulty based on the comprehension meter score
4. Escalates from questions → hints only after 2 wrong attempts
5. Responds in the student's chosen language

**Example Session:**

> 📷 *Student shows: "F = ?, m = 5 kg, a = 2 m/s²"*
>
> 🤖 *Vāṇī: "यह Newton का दूसरा नियम है! बल, द्रव्यमान और त्वरण के बीच क्या संबंध है?"*
> *(Translation: "This is Newton's Second Law! What is the relation between force, mass, and acceleration?")*
>
> 🎒 *Student: "F = m × a"*
>
> 🤖 *Vāṇī: "बिल्कुल सही! अब m = 5 और a = 2 को सूत्र में रखने पर क्या मिलेगा?"*
> *(Translation: "Exactly right! Now what do you get when you put m = 5 and a = 2 into the formula?")*

---

## 🌍 Supported Languages

| Language | Code | Script |
|---|---|---|
| English | en | Latin |
| Hindi | hi | Devanagari |
| Kannada | kn | Kannada |
| Tamil | ta | Tamil |
| Telugu | te | Telugu |

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Intel i3 (8th gen) | Intel i5 / Ryzen 5 |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB free | 10 GB SSD |
| Camera | 720p webcam | 1080p webcam |
| GPU | Not required | Not required |
| Internet | Not required | Not required |

---

## 🔮 Future Scope

- 📱 Mobile app (Android-first, Kivy / React Native)
- 🎤 Voice input/output (Whisper Tiny + TTS)
- 📚 Full curriculum (NCERT Grades 6–12)
- 🌐 15+ Indian languages
- 📊 Teacher dashboard with student progress analytics
- 🏫 School-level deployment via local network (no internet)

---

## 🏆 Innovation Highlights

- **Edge AI for Social Good** — AI education at the absolute lowest cost of ownership
- **Socratic Reasoning AI** — teaches critical thinking, not memorization
- **Offline Multimodal** — camera + text + multilingual in one lightweight package
- **CPU-Optimized** — runs on hardware already present in government schools

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
  Built with ❤️ for Bharat 🇮🇳 · Vāṇī-Vision · Empowering 500M+ offline learners
</div>
