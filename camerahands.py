import os
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from keras.models import load_model
from datetime import datetime
from django.shortcuts import render


class SignLanguageDetector:
    def __init__(self):
        self.model = load_model('smnist.h5')
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                       'V', 'W', 'X', 'Y']
        self.last_prediction_time = 0
        self.prediction_interval = 1.0  # Predict every second
        self.last_prediction = None
        
        # New variables for word formation
        self.current_word = []
        self.formed_words = []
        self.consecutive_count = 0
        self.last_letter = None
        self.last_letter_time = time.time()  # Track time of last letter detection
        self.timeout_duration = 3.0 # 2.5 seconds timeout

    def check_timeout(self, current_time):
        if self.current_word and (current_time - self.last_letter_time) >= self.timeout_duration:
            formed_word = ''.join(self.current_word)
            if formed_word:  # Only add if word is not empty
                self.formed_words.append(formed_word)
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Word completed (timeout): {formed_word}")
                print(f"All words: {' '.join(self.formed_words)}\n")
            self.current_word = []
            self.consecutive_count = 0
            self.last_letter = None

    def process_hand(self, frame, w, h):
        current_time = time.time()
        self.check_timeout(current_time)  # Check for timeout before processing

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand bounding box
                x_max = y_max = 0
                x_min, y_min = w, h
                
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max = max(x_max, x)
                    x_min = min(x_min, x)
                    y_max = max(y_max, y)
                    y_min = min(y_min, y)
                
                # Add padding
                y_min = max(0, y_min - 20)
                y_max = min(h, y_max + 20)
                x_min = max(0, x_min - 20)
                x_max = min(w, x_max + 20)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Process hand image for prediction
                if current_time - self.last_prediction_time >= self.prediction_interval:
                    hand_img = frame_gray[y_min:y_max, x_min:x_max]
                    if hand_img.size > 0:
                        hand_img = cv2.resize(hand_img, (28, 28))
                        
                        # Prepare image for prediction
                        nlist = []
                        rows, cols = hand_img.shape
                        for i in range(rows):
                            for j in range(cols):
                                nlist.append(hand_img[i, j])
                        
                        # Make prediction
                        pixel_data = np.array(nlist).reshape(-1, 28, 28, 1) / 255.0
                        prediction = self.model.predict(pixel_data, verbose=0)
                        pred_array = prediction[0]
                        
                        # Get top prediction
                        max_idx = np.argmax(pred_array)
                        confidence = pred_array[max_idx] * 100
                        letter = self.letters[max_idx]
                        
                        # Only process if confidence is above threshold
                        if confidence > 40:  # You can adjust this threshold
                            current_time_str = datetime.now().strftime("%H:%M:%S")
                            self.last_letter_time = current_time  # Update last letter time
                            
                            # Handle word formation
                            if letter == self.last_letter:
                                self.consecutive_count += 1
                                if self.consecutive_count == 2:  # Letter detected twice
                                    self.current_word.append(letter)
                                    print(f"[{current_time_str}] Added letter: {letter} to current word")
                                    print(f"Current word: {''.join(self.current_word)}")
                            else:
                                self.consecutive_count = 1
                            
                            # Check for zero (space) to end word
                            if letter == 'O' and confidence > 80:  # Using 'O' as space with higher confidence threshold
                                if self.current_word:
                                    formed_word = ''.join(self.current_word)
                                    self.formed_words.append(formed_word)
                                    print(f"\n[{current_time_str}] Word completed: {formed_word}")
                                    print(f"All words: {' '.join(self.formed_words)}\n")
                                    self.current_word = []
                            
                            self.last_letter = letter
                            self.last_prediction = (letter, confidence)
                        
                        self.last_prediction_time = current_time
                
                # Draw landmarks
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, 
                                            self.mphands.HAND_CONNECTIONS)
        
        # Add text overlay for current word and formed words
        cv2.putText(frame, f"Current word: {''.join(self.current_word)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Formed words: {' '.join(self.formed_words)}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add timeout countdown if word is being formed
        if self.current_word:
            time_left = max(0, self.timeout_duration - (current_time - self.last_letter_time))
            cv2.putText(frame, f"Timeout in: {time_left:.1f}s", 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame



    def run(self):
        print("\n=== Sign Language Detection Started ===")
        print("Show letter sign twice to add it to current word")
        print("Show 'O' sign to complete word")
        print("No input for 3 seconds will complete current word")
        print("Press ESC to exit\n")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        _, frame = cap.read()
        h, w, _ = frame.shape
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process frame
                frame = self.process_hand(frame, w, h)
                
                # Display frame
                cv2.imshow("Sign Language Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    if self.formed_words:
                        print("\n=== Final Results ===")
                        print(f"Words formed: {' '.join(self.formed_words)}")
                    print("\n=== Detection Stopped ===")
                    break
                
        except KeyboardInterrupt:
            print("\n=== Detection Interrupted ===")
        finally:
            cap.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.run()
