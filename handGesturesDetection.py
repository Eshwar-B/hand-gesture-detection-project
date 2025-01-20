import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Define some variables
drawing = False
start_point = None
thickness = 5

# Open camera feed
cap = cv2.VideoCapture(0)

# Create a blank white canvas for drawing
drawing_layer = np.zeros((480, 640, 3), dtype="uint8")

def count_fingers(hand_landmarks):
    # Check if the hand landmarks are detected and count extended fingers
    count = 0
    finger_tips = [8, 12, 16, 20]  # Index finger, middle, ring, pinky finger tips
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    # Check the thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        count += 1
    return count

# Create a resizable window
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame so that it mimics a mirror image
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count how many fingers are up
            finger_count = count_fingers(hand_landmarks)

            # Get the index fingertip coordinates
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            x = int(index_finger_tip.x * w)
            y = int(index_finger_tip.y * h)

            if finger_count == 1:  # Drawing mode (one finger)
                if not drawing:
                    start_point = (x, y)
                    drawing = True
                current_point = (x, y)
                cv2.line(drawing_layer, start_point, current_point, (0, 255, 255), thickness)  # Yellow color
                start_point = current_point
            elif finger_count == 2:  # Moving mode (two fingers)
                start_point = None
                drawing = False
            elif finger_count == 5:  # Erase mode (all fingers)
                drawing_layer = np.zeros((480, 640, 3), dtype="uint8")  # Clear the drawing layer
                drawing = False

    # Combine the drawing layer with the camera feed
    combined_frame = cv2.addWeighted(frame, 1, drawing_layer, 1, 0)

    # Display the combined frame
    cv2.imshow("Camera", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
