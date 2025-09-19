# Hand Gesture Translator 🤖✋🔤  

## 📌 Overview  
This project combines **Artificial Intelligence (AI)**, **Computer Vision (CV)**, **IoT**, and **MIT App Inventor** to create a **Hand Gesture Translator**.  

The system can:  
- Recognize **hand gesture letters** in real time.  
- Convert gestures into **words and sentences**.  
- Perform the **reverse task**: translate text into **animated hand gestures**.  

The goal is to help bridge communication gaps for people with hearing and speech impairments by providing an accessible, bi-directional translation tool.  

---

## 🚀 Features  

### 🔹 AI / Computer Vision  
- Dataset: **English letters & numbers**.  
- Implemented approaches:  
  - **ANN + MediaPipe** → fast & accurate landmark recognition.  
  - **YOLO detector + MobileNet classifier** → robust detection & classification.  
- Builds words & sentences from continuous gesture recognition.  

### 🔹 GUI (NiceGUI)  
- Clean and interactive **Graphical User Interface** built with **NiceGUI**.  
- Displays **real-time predictions** from the AI model.  

### 🔹 IoT Integration  
- **ESP microcontroller** used to transmit recognized letters.  
- Output displayed on an **LCD screen** for real-world interaction.  

### 🔹 MIT App Inventor (Reverse Translation)  
- Users can type words in English.  
- The app displays **hand gesture animations** for each letter.  
- Enables **bi-directional translation**:  
  - **Gestures ➝ Text**  
  - **Text ➝ Gestures**  

---

## 🏗️ System Architecture  
1. **Computer Vision Model** → Detects hand gestures & classifies letters.  
2. **GUI** → Displays predictions in real time.  
3. **IoT (ESP + LCD)** → Outputs recognized letters physically.  
4. **MIT App** → Converts typed words back into gesture animations.  

---

## 🛠️ Tech Stack  
- **AI/ML**: Python, Mediapipe, YOLO, MobileNet, ANN  
- **GUI**: NiceGUI  
- **IoT**: ESP microcontroller, LCD display  
- **App Development**: MIT App Inventor  

---

## 🔮 Future Improvements  
- Add support for **multiple languages**.  
- Extend to **sentence-level sign gestures**.  
- Enhance **gesture animation smoothness** in MIT App.  
- Deploy AI model on **edge devices** for offline use.  

---
