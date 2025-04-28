import cv2
import mediapipe as mp
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def recognize_gesture(hand_landmarks):
    lm = hand_landmarks.landmark
    get = lambda point: (lm[point].x, lm[point].y)

    thumb_tip     = get(mp_hands.HandLandmark.THUMB_TIP)
    thumb_ip      = get(mp_hands.HandLandmark.THUMB_IP)
    index_tip     = get(mp_hands.HandLandmark.INDEX_FINGER_TIP)
    index_mcp     = get(mp_hands.HandLandmark.INDEX_FINGER_MCP)
    middle_tip    = get(mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
    middle_mcp    = get(mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
    ring_tip      = get(mp_hands.HandLandmark.RING_FINGER_TIP)
    pinky_tip     = get(mp_hands.HandLandmark.PINKY_TIP)
    wrist         = get(mp_hands.HandLandmark.WRIST)

    if (
         index_tip[1] > index_mcp[1] and
         middle_tip[1] > middle_mcp[1] and
         ring_tip[1] > wrist[1] and
         pinky_tip[1] > wrist[1] and
         thumb_tip[1] > thumb_ip[1]
    ):
        return "fist"

    if (
        index_tip[1] < index_mcp[1] and
        middle_tip[1] < middle_mcp[1] and
        ring_tip[1] < wrist[1] and
        pinky_tip[1] < wrist[1] and
        thumb_tip[1] < wrist[1]
    ):
        return "stop"

    if (
        index_tip[1] < index_mcp[1] and
        middle_tip[1] < middle_mcp[1] and
        ring_tip[1] > wrist[1] and
        pinky_tip[1] > wrist[1]
    ):
         return "peace"

    if (
        thumb_tip[1] < thumb_ip[1] and
        index_tip[1] > index_mcp[1] and
        middle_tip[1] > middle_mcp[1]
    ):
        return "like"

    if (
        thumb_tip[1] > thumb_ip[1] and
        index_tip[1] > index_mcp[1] and
        middle_tip[1] > middle_mcp[1]
    ):
        return "dislike"

    return "unknown"

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            asl_letter = recognize_gesture(hand_landmarks)
            return asl_letter

    return "unknown"

data_folder = './data/stop'

image_files = [f for f in os.listdir(data_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

gesture_count = {
    "fist": 0,
    "stop": 0,
    "peace": 0,
    "like": 0,
    "dislike": 0,
    "unknown": 0
}

start_time = time.time()

for image_file in image_files:
    image_path = os.path.join(data_folder, image_file)
    gesture = process_image(image_path)
    if gesture:
        gesture_count[gesture] += 1

end_time = time.time()

for gesture, count in gesture_count.items():
    print(f"Gesture '{gesture}': {count} images")

elapsed_time = end_time - start_time
print(f"\nОбщее время работы алгоритма: {elapsed_time:.2f} секунд")
