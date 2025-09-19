from nicegui import ui
import paho.mqtt.client as mqtt
import threading
import cv2
import mediapipe as mp
import base64
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import pickle
import time
import win32com.client
import pythoncom

#
# shared variables
#
user_input_lock = threading.Lock()
user_input = ''
prev_prediction = ''
current_mode_lock = threading.Lock()
current_mode = 'Letters'

#
# mqtt
#
broker = "broker.hivemq.com"
port = 1883
timeout = 60
topic = "meow/psps"

client = mqtt.Client()

def mqtt_connect():
    client.connect(broker, port, timeout)
    client.loop_start()

def mqtt_loop():
    prev_input = ''
    global user_input
    
    while True:
        with user_input_lock:
            cur_input = user_input

        if prev_input != cur_input:
            prev_input = cur_input

            ret = client.publish(topic, cur_input, qos=1)
            ret.wait_for_publish()
        time.sleep(0.01)


mqtt_connect()
threading.Thread(target=mqtt_loop, daemon=True).start()

#
# model importing
#
letters_model = load_model(r"C:\Users\MoDo\Desktop\ESP\MODELS\Big_Data_ANN\Big_Model_ANN.h5")
with open(r"C:\Users\MoDo\Desktop\ESP\MODELS\Big_Data_ANN\Big_Model_Dict.pkl", "rb") as f:
    letters_dict = pickle.load(f)

numbers_model = load_model(r"C:\Users\MoDo\Desktop\ESP\MODELS\Numbers_Model\Numbers_model.h5")
with open(r"C:\Users\MoDo\Desktop\ESP\MODELS\Numbers_Model\numbers_dict.pkl", "rb") as f:
    numbers_dict = pickle.load(f)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# 
# ui web design
# 
ui.add_head_html('''
<style>
  html, body {
      margin: 0;
      padding: 0;
      background-color: #111827; /* Tailwind gray-900 */
      color: white; /* Default text color */
      height: 100%;
      width: 100%;
  }
  .light-btn {
      background-color: #f9fafb;
      color: #111827;
      border: 4px solid #080B12;
      border-radius: 1rem;
      box-shadow: 0 0 20px 4px rgba(59, 130, 246, 0.3);
      padding: 0.5rem 1.5rem;
      font-weight: 600;
      transition: all 0.2s ease-in-out;
  }
  .active-btn {
      background-color: #214D94 !important; /* Active: Blue */
      color: #C2C2C2 !important;
      border: 4px solid #111827 !important;
      box-shadow: 0 0 25px 5px rgba(59, 130, 246, 0.5);
  }
</style>
''')

#
# ui helpers
#
def frame_to_data_url(bgr_frame, quality=70):
    ret, buf = cv2.imencode('.jpg', bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        return None
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('ascii')

def updateText():
    with user_input_lock:
        label.set_text(user_input)

def toggle_buttons(selected):
    global current_mode
    current_mode = selected

    if selected == 'Letters':
        letters_button.classes(remove='active-btn').classes(add='active-btn')
        numbers_button.classes(remove='active-btn')
    else:
        numbers_button.classes(remove='active-btn').classes(add='active-btn')
        letters_button.classes(remove='active-btn')

def clear_input():
    global user_input
    with user_input_lock:
        user_input = ""
        label.set_text("")

camera_status = "on"

def camera_toggle():
    global camera_status
    if camera_status == 'on':
        camera_status = "off"
    else:
        camera_status = "on"

def tts_speak():
    
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Rate = -1
    speaker.Volume = 100

    global user_input
    with user_input_lock:
        word_to_speak = user_input

    speaker.Speak(word_to_speak)
    pythoncom.CoUninitialize()

def tts_wrapper():
    threading.Thread(target=tts_speak, daemon=True).start()

#
# ui
#
with ui.column().classes('w-full h-screen justify-center items-center gap-10'):
    placeholder = ("data:image/png;base64,"
                   "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg"
                   "YAAAAAMAAWgmWQ0AAAAASUVORK5CYII=")

    with ui.row().classes('justify-center items-center gap-20'):
        camera_toggle_button = ui.button("CAMERA TOGGLE", on_click=lambda: camera_toggle()).classes('light-btn')
        
        webcam = ui.interactive_image(placeholder).classes('w-[40vw] object-cover'
                ).style('box-shadow: 0 0 30px 6px rgba(59,130,246,0.6); border: 4px solid #111827;')
        
        tts_button = ui.button("Text-To-Speech", on_click=lambda: tts_wrapper()).classes('light-btn')

    with ui.row().classes('justify-center gap-4'):
        letters_button = ui.button('Letters', on_click=lambda: toggle_buttons('Letters')).classes('light-btn')
        numbers_button = ui.button('Numbers', on_click=lambda: toggle_buttons('Numbers')).classes('light-btn')
    
    label = ui.label('').classes('text-2xl font-bold text-center')

    clear_button = ui.button('CLEAR', on_click=lambda: clear_input()).classes('light-btn')


letters_button.classes(add='active-btn')

#
# model run
#
predictions = {}
prediction_counter = 0
def run_prediction():
    global camera_status
    global predictions
    global prediction_counter
    global user_input
    global prev_prediction
    predicted_label = ""

    ret, frame = cap.read()
    
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if camera_status == "off" or not result.multi_hand_landmarks:
        if 'NONE' not in predictions:
            predictions['NONE'] = 0
        predictions['NONE'] += 1
    else:
        for hand_landmarks in result.multi_hand_landmarks:
            x, y_, features = [], [], []

            for lm in hand_landmarks.landmark:
                x.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                features.append(lm.x - min(x))
                features.append(lm.y - min(y_))

            if len(features) == 42:
                features = np.asarray(features).reshape(1, -1)

                if current_mode == 'Letters':
                    prediction = letters_model.predict(features, verbose=0)
                    pred_index = int(np.argmax(prediction))
                    predicted_label = str(letters_dict[pred_index])
                else:
                    prediction = numbers_model.predict(features, verbose=0)
                    pred_index = int(np.argmax(prediction))
                    predicted_label = str(numbers_dict[pred_index])

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if predicted_label not in predictions:
                    predictions[predicted_label] = 0
                predictions[predicted_label] += 1
    
    prediction_counter += 1
    if prediction_counter > 10:
        max_prediction_key = max(predictions, key=predictions.get)
        max_prediction_val = max(predictions.values())
        
        if max_prediction_key != "NONE" and max_prediction_val > 7:
            with user_input_lock:
                user_input += predicted_label
            prev_prediction = predicted_label

        prediction_counter = 0
        predictions.clear()


    if camera_status == "on":
        cv2.putText(frame, "CAMERA ON", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "CAMERA OFF", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    data_url = frame_to_data_url(frame)
    if data_url:
        webcam.source = data_url
        webcam.update()

ui.timer(0.01, run_prediction)
ui.timer(0.2, lambda: updateText())

ui.run(host='0.0.0.0', port=8081, reload=False)
