import cv2
import mediapipe as mp
import math

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

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            asl_letter = recognize_gesture(hand_landmarks)
            cv2.putText(frame, f"Detected: {asl_letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

    cv2.imshow('ASL Hand Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
