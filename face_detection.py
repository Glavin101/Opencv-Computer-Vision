import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

while True:
    _, image = video.read()
    image = cv2.flip(image, 1)  # Flip the image

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
    cv2.imshow('Face Detection', image)

    # Detect escape key press to exit the loop
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()


