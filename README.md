Hand Gesture Translator 🤖✋🔤
Overview

This project combines Artificial Intelligence (AI), Computer Vision (CV), IoT, and MIT App Inventor to create a Hand Gesture Translator.
The system can recognize hand gesture letters in real time, convert them into words and sentences, and also perform the reverse task—translating text back into animated hand gestures.

The aim is to bridge the gap in communication for people with hearing and speech impairments by providing an accessible and interactive translation tool.

Features
🔹 AI / Computer Vision

Used datasets of English letters and numbers.

Implemented multiple approaches:

ANN with MediaPipe → fast and accurate hand landmark recognition.

YOLO detector + MobileNet classifier → robust detection and classification pipeline.

Generates words and sentences by recognizing sequences of hand gestures.

🔹 GUI (NiceGUI)

Built an intuitive Graphical User Interface (GUI) using NiceGUI.

Displays real-time predictions from the AI model.

🔹 IoT Integration

Used an ESP microcontroller to send recognized letters from the AI model.

Output is displayed on an LCD screen, enabling physical interaction and real-world deployment.

🔹 MIT App Inventor (Reverse Translation)

A custom-built MIT App allows users to input words in English.

The app then displays corresponding hand gesture animations, simulating sign language movements.

This makes the system bi-directional:

Gestures ➝ Text

Text ➝ Gestures

System Architecture

Computer Vision Model → Detects hand gestures & classifies letters.

GUI → Shows real-time predictions.

IoT Module (ESP + LCD) → Displays letters physically.

MIT App → Converts English words back into gesture animations.

Tech Stack

AI/ML: Python, Mediapipe, YOLO, MobileNet, ANN

GUI: NiceGUI

IoT: ESP microcontroller, LCD display

App Development: MIT App Inventor

Future Improvements

Extend support for more languages.

Add sentence-level sign language gestures.

Improve gesture animation smoothness in the MIT App.

Deploy AI model on edge devices for offline use.
