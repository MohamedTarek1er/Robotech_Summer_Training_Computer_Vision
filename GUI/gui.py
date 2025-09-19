from nicegui import ui, app
import paho.mqtt.client as mqtt
import threading
import cv2
import mediapipe as mp
import base64
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pickle
import time
import win32com.client
import pythoncom
import sys
import os

#
# shared variables
#
application_running = True
user_input_lock = threading.Lock()
user_input = ''
current_mode_lock = threading.Lock()
current_mode = 'Letters'
camera_status_lock = threading.Lock()
camera_status = "on"
webcam_frame_lock = threading.Lock()
webcam_frame = ""

#
# mqtt
#
def mqtt_thread():
    broker = "broker.hivemq.com"
    port = 1883
    timeout = 60
    topic = "meow/psps"

    client = mqtt.Client()
    client.connect(broker, port, timeout)
    client.loop_start()

    prev_input = ''
    global user_input
    global application_running
    
    while application_running:
        with user_input_lock:
            cur_input = user_input

        if prev_input != cur_input:
            prev_input = cur_input

            ret = client.publish(topic, cur_input, qos=1)
            ret.wait_for_publish()
        time.sleep(0.01)

threading.Thread(target=mqtt_thread, daemon=True).start()

#
# model prediction
#
def prediction_thread():
    if not os.path.exists("tf_lite_models/letters_model.tflite"):
        letters_model_keras = load_model("keras_models/letters_model.h5")
        tflite_model = tf.lite.TFLiteConverter.from_keras_model(letters_model_keras).convert()
        with open("tf_lite_models/letters_model.tflite", "wb") as f: f.write(tflite_model)

    letters_model = tf.lite.Interpreter(model_path="tf_lite_models/letters_model.tflite")
    letters_model.allocate_tensors()
    letters_model_input = letters_model.get_input_details()
    letters_model_output = letters_model.get_output_details()

    with open("dictionaries/letters_dict.pkl", "rb") as f: letters_dict = pickle.load(f)

    if not os.path.exists("tf_lite_models/numbers_model.tflife"):
        numbers_model_keras = load_model("keras_models/numbers_model.h5")
        tflite_model = tf.lite.TFLiteConverter.from_keras_model(numbers_model_keras).convert()
        with open("tf_lite_models/numbers_model.tflife", "wb") as f: f.write(tflite_model)

    numbers_model = tf.lite.Interpreter(model_path="tf_lite_models/numbers_model.tflife")
    numbers_model.allocate_tensors()
    numbers_model_input = numbers_model.get_input_details()
    numbers_model_output = numbers_model.get_output_details()

    with open("dictionaries/numbers_dict.pkl", "rb") as f: numbers_dict = pickle.load(f)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, model_complexity=1)
    mp_draw = mp.solutions.drawing_utils
    landmark_point_color = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
    landmark_connection_color = mp_draw.DrawingSpec(color=(0, 150, 0), thickness=2)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    global user_input
    global camera_status
    global current_mode
    global webcam_frame
    global application_running

    predictions = {}
    predictions_counter = 0
    predictions_interval = 0.5
    prev_prediction_time = time.time()

    while application_running:
        ret, frame = cap.read()
        
        if not ret:
            time.sleep(0.001)
            continue

        with camera_status_lock: camera_status_local = camera_status
        with current_mode_lock: current_mode_local = current_mode

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)


        if camera_status_local == "off" or not result.multi_hand_landmarks:
            if 'NONE' not in predictions:
                predictions['NONE'] = 0
            predictions['NONE'] += 1
        else:
            for hand_landmarks in result.multi_hand_landmarks:
                x_min = sys.float_info.max
                y_min = sys.float_info.max
                
                for lm in hand_landmarks.landmark:
                    x_min = min(x_min, lm.x)
                    y_min = min(y_min, lm.y)

                features = []
                for lm in hand_landmarks.landmark:
                    features.append(lm.x - x_min)
                    features.append(lm.y - y_min)

                if len(features) == 42:
                    features = np.asarray(features, dtype=np.float32).reshape(1, -1)
                    predicted_label = ""

                    if current_mode_local == 'Letters':
                        letters_model.set_tensor(letters_model_input[0]['index'], features)
                        letters_model.invoke()
                        prediction = letters_model.get_tensor(letters_model_output[0]['index'])
                        pred_index = int(np.argmax(prediction))
                        predicted_label = str(letters_dict[pred_index])
                    else:
                        numbers_model.set_tensor(numbers_model_input[0]['index'], features)
                        numbers_model.invoke()
                        prediction = numbers_model.get_tensor(numbers_model_output[0]['index'])
                        pred_index = int(np.argmax(prediction))
                        predicted_label = str(numbers_dict[pred_index])

                    if predicted_label not in predictions:
                        predictions[predicted_label] = 0
                    predictions[predicted_label] += 1

                    mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS, 
                        landmark_drawing_spec=landmark_point_color,
                        connection_drawing_spec=landmark_connection_color)
        

        predictions_counter += 1
        cur_time = time.time()
        intervalPassed = cur_time - prev_prediction_time > predictions_interval
        
        if predictions_counter > 10 and intervalPassed:
            max_prediction_key = max(predictions, key=predictions.get)
            max_prediction_val = predictions[max_prediction_key]
            prediction_threshold = predictions_counter * 0.7
            
            if max_prediction_key != "NONE" and max_prediction_val > prediction_threshold:
                with user_input_lock:
                    user_input += max_prediction_key
            
            predictions_counter = 0
            predictions.clear()
            prev_prediction_time = cur_time


        if camera_status == "on":
            cv2.putText(frame, "CAMERA ON", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "CAMERA OFF", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        

        webcam_frame_local = frame_to_data_url(frame, 80)
        with webcam_frame_lock:
            webcam_frame = webcam_frame_local

threading.Thread(target=prediction_thread, daemon=True).start()

#
# ui helpers
#
def frame_to_data_url(bgr_frame, quality=70):
    ret, buf = cv2.imencode('.jpg', bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        return None
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('ascii')

def updateWebcam():
    with webcam_frame_lock:
        webcam_frame_local = webcam_frame
    
    if webcam_frame_local:
        webcam.source = webcam_frame_local
        webcam.update()

def updateText():
    with user_input_lock:
        label.set_text(user_input)

def toggle_buttons(selected):
    global current_mode
    with current_mode_lock:
        current_mode = selected

    letters_button.classes(remove='active-btn')
    numbers_button.classes(remove='active-btn')

    if selected == "Letters":
        letters_button.classes(add='active-btn')
    else:
        numbers_button.classes(add='active-btn')

def clear_input():
    global user_input
    with user_input_lock:
        user_input = ""
        label.set_text("")

def camera_toggle():
    global camera_status
    with camera_status_lock:
        if camera_status == "on":
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

def application_exit():
    global application_running
    application_running = False
    app.shutdown()

# 
# ui web design
# 
ui.add_head_html('''
<style>
  html, body {
      margin: 0;
      padding: 0;
      background-color: #111827;
      color: white;
      height: 100%;
      width: 100%;
      scrollbar-width: none;
      -ms-overflow-style: none;
  }
  body::-webkit-scrollbar {
      display: none;
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
      background-color: #214D94 !important;
      color: #C2C2C2 !important;
      border: 4px solid #111827 !important;
      box-shadow: 0 0 25px 5px rgba(59, 130, 246, 0.5);
  }
</style>
''')

#
# ui
#
with ui.column().classes('w-full h-screen justify-center items-center gap-10'):
    with ui.row().classes('justify-center items-center gap-8'):
        exit_button = ui.button('EXIT', on_click=lambda:application_exit()).classes('light-btn')

    placeholder = ("data:image/png;base64,"
                   "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg"
                   "YAAAAAMAAWgmWQ0AAAAASUVORK5CYII=")

    with ui.row().classes('justify-center items-center gap-20'):
        camera_toggle_button = ui.button("CAMERA TOGGLE", on_click=lambda: camera_toggle()).classes('light-btn')
        
        webcam = ui.interactive_image(placeholder).classes('w-[40vw] object-cover'
                ).style('box-shadow: 0 0 30px 6px rgba(59,130,246,0.6); border: 4px solid #111827;')
        
        tts_button = ui.button("Text-To-Speech", on_click=lambda: tts_wrapper()).classes('light-btn')

    with ui.row().classes('justify-center gap-4'):
        letters_button = ui.button('Letters', on_click=lambda: toggle_buttons('Letters')).classes('light-btn').classes('active-btn')
        numbers_button = ui.button('Numbers', on_click=lambda: toggle_buttons('Numbers')).classes('light-btn')
    
    label = ui.label('').classes('text-2xl font-bold text-center')

    clear_button = ui.button('CLEAR', on_click=lambda: clear_input()).classes('light-btn')

#
# periodic updates
#
ui.timer(0.01, updateText)
ui.timer(0.01, updateWebcam)

#
# main
#
ui.run(reload=False, fullscreen=True)
