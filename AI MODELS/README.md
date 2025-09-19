## 🤖 AI Models  

The AI part of this project was the core component for recognizing **hand gestures** and mapping them to **English letters and numbers**.  
We experimented with different datasets and architectures to evaluate accuracy and efficiency.  

---

### 📊 Datasets  
- **Large Dataset (Big Data):** ~3000 images per letter (A–Z).  
- **Small Dataset:** A reduced version used for faster experimentation.  
- **Numbers Dataset:** ~1000 images per number (1–10).  

---

### 🧪 Experiments  

#### 1️⃣ ANN + MediaPipe  
- MediaPipe used to extract **hand landmarks**.  
- ANN trained on both **small** and **large datasets**.  
- Achieved fast and reliable classification, especially on larger dataset.  

#### 2️⃣ YOLO (Detector) + MobileNet (Classifier)  
- YOLO used to **detect the hand region** in the frame.  
- MobileNet then classifies the cropped region into letters.  
- Tested on both **small** and **large datasets**.  
- Provided robustness against background noise and different hand positions.  

#### 3️⃣ MobileNet + MediaPipe  
- MediaPipe landmarks used as input to MobileNet.  
- Combined strengths of lightweight detection + deep classification.  

#### 4️⃣ Numbers Model (1–10)  
- Dataset of **1000 images per number**.  
- Used **ANN + MediaPipe landmarks**.  
- Achieved reliable classification for numerical gestures.  

---

### ⚖️ Summary  
- **Large datasets** improved model robustness and accuracy.  
- **ANN + MediaPipe** → lightweight and effective for both letters & numbers.  
- **YOLO + MobileNet** → more computationally expensive but more resilient to noise.  
- **Hybrid approach** (MobileNet + MediaPipe) showed promise in balancing speed and accuracy.  
