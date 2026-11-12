# Multilingual OCR Text Extraction, Enhancement & Speech Synthesis

A modular computer vision and NLP pipeline that preprocesses document images, recognizes multilingual text using EasyOCR, translates the extracted text across languages, and converts output into synthesized speech.

## Features
- **Adaptive Image Preprocessing**: Denoising, contrast enhancement, and thresholding with OpenCV.
- **Multilingual OCR**: Accurate text localization and character recognition via EasyOCR.
- **Translation Engine**: Translates detected text into user-selected target languages.
- **Text-to-Speech (TTS)**: Synthesizes translated text into natural audio using gTTS / pyttsx3.

## Quickstart

`ash
# Clone repository
git clone https://github.com/shamjoes/TEXT-EXTRACTION-AND-ENHANCEMENT-SYSTEM-ARCHITECTURE-EASY-OCR-AND-OPENCV.git
cd TEXT-EXTRACTION-AND-ENHANCEMENT-SYSTEM-ARCHITECTURE-EASY-OCR-AND-OPENCV

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
`

## Tech Stack
- Python 3.8+
- OpenCV
- EasyOCR
- Google Translate API / deep-translator
- pyttsx3 / gTTS