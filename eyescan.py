import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the cascade classifiers for different facial features
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
mouth_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
ear_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_leftear.xml'
eyebrow_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_rightear.xml'
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Load Haar cascades and check if they are loaded correctly
def load_cascade(cascade_path):
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"Error: Cascade file {cascade_path} not found or failed to load.")
    return cascade

eye_cascade = load_cascade(eye_cascade_path)
mouth_cascade = load_cascade(mouth_cascade_path)
ear_cascade = load_cascade(ear_cascade_path)
eyebrow_cascade = load_cascade(eyebrow_cascade_path)
face_cascade = load_cascade(face_cascade_path)

def draw_label(frame, text, rect, color=(0, 255, 0)):
    (x, y, w, h) = rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces to help locate other features relative to the face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (fx, fy, fw, fh) in faces:
        face_roi = gray[fy:fy+fh, fx:fx+fw]

        # Detect eyes within the face ROI
        if not eye_cascade.empty():
            eyes = eye_cascade.detectMultiScale(face_roi, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                draw_label(frame, 'Eye', (fx + ex, fy + ey, ew, eh))

        # Detect mouth within the face ROI
        if not mouth_cascade.empty():
            mouth = mouth_cascade.detectMultiScale(face_roi, 1.3, 20)
            for (mx, my, mw, mh) in mouth:
                if fy + my + mh > fy + fh * 1.0:  # Make sure the detected mouth is below the nose level
                    draw_label(frame, 'Mouth', (fx + mx, fy + my, mw, mh))

        # Detect ears within the face ROI
        if not ear_cascade.empty():
            ears = ear_cascade.detectMultiScale(face_roi, 1.3, 5)
            for (erx, ery, erw, erh) in ears:
                draw_label(frame, 'Ear', (fx + erx, fy + ery, erw, erh))

        # Detect eyebrows within the face ROI
        if not eyebrow_cascade.empty():
            eyebrows = eyebrow_cascade.detectMultiScale(face_roi, 1.3, 5)
            for (ebx, eby, ebw, ebh) in eyebrows:
                draw_label(frame, 'Eyebrow', (fx + ebx, fy + eby, ebw, ebh))

        # Assume hair is the top part of the face ROI (simple assumption)
        hair_rect = (fx, fy, fw, int(fh * 0.3))
        draw_label(frame, 'Hair', hair_rect)

    # Display the frame with labeled facial features
    cv2.imshow('Face Scanner', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
