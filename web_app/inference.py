import cv2
import numpy as np
import tensorflow as tf
from openvino.runtime import Core
from collections import deque
import time
import os
import threading

class AnomalyDetector:
    def __init__(self, model_dir='../Models'):
        self.model_dir = model_dir
        self.img_size = (299, 224) # Width, Height as per notebook but check Inception input
        # Note: InceptionV3 standard is 299x299, but we follow notebook config
        self.max_seq_length = 32
        self.class_vocab = ['Abuse', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting']
        
        self.lock = threading.Lock()
        
        self.load_models()
        self.processing_times = deque(maxlen=200)
        self.current_status = {'label': 'Normal', 'probability': 0.0, 'timestamp': 0}

    def load_models(self):
        print("Loading models...")
        # OpenVINO Encoder
        ie = Core()
        model_xml = os.path.join(self.model_dir, "inceptionv3_model_ir", "saved_model.xml")
        model_ir = ie.read_model(model=model_xml)
        self.compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")
        self.output_layer_ir = self.compiled_model_ir.output(0)
        
        # Keras Decoder
        decoder_path = os.path.join(self.model_dir, "classifier_lstm_e19.h5")
        self.decoder = tf.keras.models.load_model(decoder_path, compile=False)
        print("Models loaded successfully.")

    def preprocess_frame(self, frame):
        # Resize and convert BGR to RGB
        preprocessed = cv2.resize(frame, self.img_size)
        preprocessed = preprocessed[:, :, [2, 1, 0]] # BGR to RGB
        return preprocessed

    def display_text(self, frame, text, index, color=(0, 255, 0)):
        font_style = cv2.FONT_HERSHEY_DUPLEX
        font_size = 0.6
        text_vertical_interval = 25
        text_left_margin = 15
        
        text_loc = (text_left_margin, text_vertical_interval * (index + 1))
        text_loc_shadow = (text_left_margin + 1, text_vertical_interval * (index + 1) + 1)
        
        cv2.putText(frame, text, text_loc_shadow, font_style, font_size, (0, 0, 0), 1)
        cv2.putText(frame, text, text_loc, font_style, font_size, color, 1)
        return frame

    def process_video(self, video_source):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error opening video source: {video_source}")
            return

        encoder_output = []
        decoded_labels = ['N/A'] * 3
        decoded_top_probs = [0.0] * 3
        counter = 0
        final_inf_counter = 0
        
        print("Starting processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            counter += 1
            
            # Process every other frame
            if counter % 2 == 0:
                start_time = time.time()
                preprocessed = self.preprocess_frame(frame)
                
                # Encoder Inference
                try:
                    with self.lock:
                        input_tensor = preprocessed[None, ...]
                        features = self.compiled_model_ir([input_tensor])[self.output_layer_ir][0]
                    encoder_output.append(features)
                except Exception as e:
                    print(f"Encoder error: {e}")
                    continue

                if len(encoder_output) == self.max_seq_length:
                    try:
                        encoder_output_array = np.array(encoder_output)[None, ...]
                        with self.lock:
                            probabilities = self.decoder.predict(encoder_output_array, verbose=0)[0]
                        
                        # Get top 3 predictions
                        top_indices = np.argsort(probabilities)[::-1][:3]
                        for idx, i in enumerate(top_indices):
                            decoded_labels[idx] = self.class_vocab[i]
                            decoded_top_probs[idx] = probabilities[i]
                        
                        # Update current status for API
                        self.current_status = {
                            'label': decoded_labels[0],
                            'probability': float(decoded_top_probs[0]),
                            'timestamp': time.time()
                        }

                        encoder_output = [] # Reset buffer
                        final_inf_counter += 1
                    except Exception as e:
                        print(f"Decoder error: {e}")
                        encoder_output = []

                process_time = (time.time() - start_time) * 1000
                self.processing_times.append(process_time)
            
            # Visualization
            display_frame = cv2.resize(frame, (640, 480))
            
            # Draw predictions
            for i in range(3):
                text = f"{decoded_labels[i]}: {decoded_top_probs[i]*100:.1f}%"
                color = (0, 0, 255) if decoded_labels[i] != 'Normal' and decoded_top_probs[i] > 0.5 and i == 0 else (255, 255, 255)
                self.display_text(display_frame, text, i, color)

            avg_time = np.mean(self.processing_times) if self.processing_times else 0
            self.display_text(display_frame, f"Inference Time: {avg_time:.1f}ms", 3)
            self.display_text(display_frame, f"Count: {final_inf_counter}", 4)

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()
