import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
video = cv2.VideoCapture(0)

while True:
    _, image = video.read()
    image = cv2.flip(image, 1)  # Flip the image

    # Convert the image to grayscale for feature detection
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)

    # Confidence for deteciton
    confidence = 0

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(image, center, axes, 0, 0, 360, (0, 0, 255), 5)
        roi_gray = gray[y : y + h, x : x + w]

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor = 1.8, minNeighbors = 30, minSize = (25, 25))

        # Draw outlines around detected smiles
        for (sx, sy, sw, sh) in smiles:
            smile_center = (x + sx + sw // 2, y + sy + sh // 2)
            smile_axes = (sw // 2, sh // 2)
            cv2.ellipse(image, smile_center, smile_axes, 0, 0, 360, (0, 255, 0), 3)

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=30, minSize=(25, 25))

        # Draw circles around detected eyes
        for (ex, ey, ew, eh) in eyes:
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            #radius = int(round((ew + eh) * 0.25))
            #cv2.circle(image, eye_center, radius, (255, 0, 0), 5)
            eye_axes = (ew // 2, eh // 2)
            cv2.ellipse(image, eye_center, eye_axes, 0, 0, 360, (255, 0, 0), 3)

    cv2.imshow('Face Detection', image)

    # Detect escape key press to exit the loop
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
