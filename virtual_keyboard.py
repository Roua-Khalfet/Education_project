import cv2
import mediapipe as mp
import numpy as np
import cvzone
from time import sleep
from pygame import mixer
from pynput.keyboard import Controller

# Initialize pygame mixer for sound feedback
mixer.init()
try:
    feedback_sound = mixer.Sound("click.wav")
except:
    print("Sound file not found - continuing without sound feedback")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cv2.namedWindow("Eye-Controlled Virtual Keyboard", cv2.WINDOW_FULLSCREEN)

# Enhanced key layout with common words and phrases
keys = [
    ["I", "YOU", "YES", "NO", "HELLO", "THANKS"],
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["Z", "X", "C", "V", "B", "N", "M", ".", "?"],
    ["SPACE", "DEL", "CLEAR", "PREDICT", "EXIT"]
]

# Word prediction database
common_words = {
    'H': ['HELLO', 'HOW', 'HELP'],
    'T': ['THE', 'THANK', 'TIME'],
    'W': ['WHAT', 'WHEN', 'WHERE'],
    'I': ['I AM', 'I NEED', 'I WANT'],
    'P': ['PLEASE', 'POSSIBLE', 'PAIN']
}

def get_eye_position(face_landmarks, img_shape):
    if face_landmarks is None:
        return None

    # Get eye landmarks
    left_eye = np.mean([(face_landmarks.landmark[33].x * img_shape[1],
                         face_landmarks.landmark[33].y * img_shape[0]),
                        (face_landmarks.landmark[133].x * img_shape[1],
                         face_landmarks.landmark[133].y * img_shape[0])], axis=0)
    
    right_eye = np.mean([(face_landmarks.landmark[362].x * img_shape[1],
                         face_landmarks.landmark[362].y * img_shape[0]),
                         (face_landmarks.landmark[263].x * img_shape[1],
                          face_landmarks.landmark[263].y * img_shape[0])], axis=0)
    
    # Calculate midpoint between eyes
    midpoint = np.mean([left_eye, right_eye], axis=0).astype(int)
    
    return midpoint, left_eye.astype(int), right_eye.astype(int)

def detect_blink(face_landmarks, img_shape):
    if face_landmarks is None:
        return False

    # Get left eye landmarks for blink detection
    left_eye_top = face_landmarks.landmark[159].y * img_shape[0]
    left_eye_bottom = face_landmarks.landmark[145].y * img_shape[0]
    left_eye_height = abs(left_eye_top - left_eye_bottom)

    # Get right eye landmarks for blink detection
    right_eye_top = face_landmarks.landmark[386].y * img_shape[0]
    right_eye_bottom = face_landmarks.landmark[374].y * img_shape[0]
    right_eye_height = abs(right_eye_top - right_eye_bottom)

    # Threshold for blink detection
    blink_threshold = 0.015 * img_shape[0]
    return left_eye_height < blink_threshold and right_eye_height < blink_threshold

class Button:
    def __init__(self, pos, text, size=[70, 70], color=(255, 0, 255)):
        self.pos = pos
        self.size = size
        self.original_color = color
        self.current_color = color
        if text in ["SPACE", "DEL", "CLEAR", "PREDICT", "EXIT"]:
            self.size = [150, 70]
        elif text in ["I", "YOU", "YES", "NO", "HELLO", "THANKS"]:
            self.size = [120, 70]
            self.original_color = (0, 200, 150)
        self.text = text
        self.hover_time = 0
        self.activation_threshold = 15

    def draw(self, img, hover_progress=0):
        x, y = self.pos
        w, h = self.size
        
        if hover_progress > 0:
            alpha = min(hover_progress / self.activation_threshold, 1.0)
            self.current_color = tuple(map(lambda x, y: int(x + (y-x)*alpha), 
                                        self.original_color, (0, 255, 0)))
        
        cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.current_color, cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7 if len(self.text) > 5 else 0.9
        text_size = cv2.getTextSize(self.text, font, font_scale, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        
        cv2.putText(img, self.text, (text_x+2, text_y+2), font, font_scale, (0, 0, 0), 2)
        cv2.putText(img, self.text, (text_x, text_y), font, font_scale, (255, 255, 255), 2)

def create_buttons():
    buttonList = []
    for j, key in enumerate(keys[0]):
        buttonList.append(Button([180 * j + 50, 50], key))
    
    for i in range(1, 4):
        for j, key in enumerate(keys[i]):
            buttonList.append(Button([90 * j + 50, 90 * i + 100], key))
    
    for j, key in enumerate(keys[4]):
        buttonList.append(Button([180 * j + 50, 450], key))
    
    return buttonList

def update_predictions(text):
    if not text:
        return []
    last_word = text.split()[-1] if text.split() else text
    first_char = last_word[0].upper()
    return common_words.get(first_char, [])[:3]

def draw_text_area(img, text, predictions):
    cv2.rectangle(img, (50, 600), (1200, 680), (175, 0, 175), cv2.FILLED)
    if len(text) > 40:
        displayed_text = "..." + text[-40:]
    else:
        displayed_text = text
    cv2.putText(img, displayed_text, (60, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if predictions:
        cv2.rectangle(img, (50, 520), (1200, 580), (0, 100, 0), cv2.FILLED)
        pred_text = " | ".join(predictions)
        cv2.putText(img, pred_text, (60, 560), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def main():
    buttonList = create_buttons()
    text_memory = []
    last_button = None
    keyboard = Controller()
    last_blink_time = 0
    blink_cooldown = 0.5  # Seconds between blinks
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to get frame from camera")
                break
                
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            predictions = update_predictions(''.join(text_memory))
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                eye_data = get_eye_position(face_landmarks, img.shape)
                
                if eye_data:
                    midpoint, left_eye, right_eye = eye_data
                    
                    # Draw eyes and midpoint
                    cv2.circle(img, tuple(left_eye), 3, (0, 255, 0), -1)
                    cv2.circle(img, tuple(right_eye), 3, (0, 255, 0), -1)
                    cv2.circle(img, tuple(midpoint), 5, (255, 0, 0), -1)
                    
                    # Check for blink
                    is_blinking = detect_blink(face_landmarks, img.shape)
                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                    
                    for button in buttonList:
                        x, y = button.pos
                        w, h = button.size

                        if x < midpoint[0] < x + w and y < midpoint[1] < y + h:
                            if last_button == button:
                                button.draw(img, button.hover_time)
                                if is_blinking and (current_time - last_blink_time) > blink_cooldown:
                                    try:
                                        mixer.Sound.play(feedback_sound)
                                    except:
                                        pass
                                        
                                    if button.text == "SPACE":
                                        text_memory.append(" ")
                                    elif button.text == "DEL":
                                        if text_memory: text_memory.pop()
                                    elif button.text == "CLEAR":
                                        text_memory = []
                                    elif button.text == "EXIT":
                                        raise KeyboardInterrupt
                                    elif button.text == "PREDICT":
                                        if predictions:
                                            text_memory.extend(list(predictions[0]))
                                    else:
                                        text_memory.extend(list(button.text))
                                    
                                    last_blink_time = current_time
                            last_button = button
                        else:
                            button.draw(img, 0)
                            if button == last_button:
                                button.hover_time = 0
            else:
                for button in buttonList:
                    button.draw(img, 0)

            finalText = ''.join(text_memory)
            draw_text_area(img, finalText, predictions)
            # Afficher le texte dans le terminal
            print("Text entered: ", finalText)
            cv2.putText(img, "Look at letters and blink to select", (50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Eye-Controlled Virtual Keyboard", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Exiting program...")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()