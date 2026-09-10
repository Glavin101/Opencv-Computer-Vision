import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the pre-trained hand landmark model for hand detection
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options)
hand_detection = vision.HandLandmarker.create_from_options(options)

video = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
x1 = x2 = y1 = y2 = 0

while True:
    _, image = video.read()
    image = cv2.flip(image, 1)  # Flip the image
    image_height, image_width, _ = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=gray)
    hand_output = hand_detection.detect(mp_image)
    all_hands = hand_output.hand_landmarks

    if all_hands:
        for hand in all_hands:
            one_hand_landmarks = hand

            for id, lm in enumerate(one_hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                if id == 8:  # Index finger tip
                    # Calculate the mouse cursor position based on the index finger tip position
                    mouse_x = int(screen_width / image_width * x)
                    mouse_y = int(screen_height / image_height * y)

                    cv2.circle(image, (x, y), 10, (0, 255, 0), cv2.FILLED)
                    pyautogui.moveTo(mouse_x, mouse_y)  # Move the mouse cursor

                    x1 = x
                    y1 = y

                if id == 4: # Thumb tip
                    x2 = x
                    y2 = y
                    cv2.circle(image, (x, y), 10, (0, 255, 0), cv2.FILLED)

            d = y2 - y1  # Calculate the distance between the index finger tip and thumb tip
            if d < 20:
                pyautogui.click()  # Perform a mouse click

    cv2.imshow('Hand movement detection', image)

    # Detect escape key press to exit the loop
    key = cv2.waitKey(10)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()