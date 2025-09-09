import cv2
import numpy as np
import mediapipe as mp  # Replace dlib with MediaPipe
from tensorflow.keras.models import load_model

# Load your custom CNN model (unchanged)
model_path = r"C:\Users\tariq\Desktop\Try\eye_detection_light_model.keras"
model = load_model(model_path)

# Initialize MediaPipe Face Mesh (for eye landmarks)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # Better for live video
    max_num_faces=1,         # Focus on the driver
    refine_landmarks=True,   # Adds iris landmarks (better precision)
)

# MediaPipe's eye landmark indices (left and right eyes)
# These are predefined indices for the eye contours in the 468-point face mesh.
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Outer and inner corners
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with MediaPipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]  # Get the first face

        # Function to extract and crop eye region
        def get_eye_region(eye_indices, frame):
            # Get eye landmark coordinates
            eye_points = np.array([(int(landmarks.landmark[i].x * frame.shape[1]), 
                                  int(landmarks.landmark[i].y * frame.shape[0])) 
                                 for i in eye_indices])
            
            # Get bounding box with padding
            x, y, w, h = cv2.boundingRect(eye_points)
            padding = 10  # Expand the bounding box slightly
            x, y = max(0, x - padding), max(0, y - padding)
            w, h = min(frame.shape[1] - x, w + 2 * padding), min(frame.shape[0] - y, h + 2 * padding)
            
            # Crop and return eye region
            eye_region = frame[y:y+h, x:x+w]
            return eye_region, (x, y, w, h)

        # Get left and right eye regions
        left_eye_region, left_rect = get_eye_region(LEFT_EYE_INDICES, frame)
        right_eye_region, right_rect = get_eye_region(RIGHT_EYE_INDICES, frame)

        # Preprocess eye regions for CNN (same as your original code)
        def preprocess_eye(eye_region):
            if eye_region.size == 0:
                return None  # Skip if no eye detected
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            resized_eye = cv2.resize(gray_eye, (24, 24))
            normalized_eye = resized_eye / 255.0
            return np.expand_dims(normalized_eye, axis=(0, -1))  # Add batch and channel dims

        left_eye_input = preprocess_eye(left_eye_region)
        right_eye_input = preprocess_eye(right_eye_region)

        # Predict eye state (if eyes are detected)
        left_eye_status = "Unknown"
        right_eye_status = "Unknown"
        
        if left_eye_input is not None:
            left_eye_pred = model.predict(left_eye_input)
            left_eye_status = "Open" if np.argmax(left_eye_pred) == 1 else "Closed"
        
        if right_eye_input is not None:
            right_eye_pred = model.predict(right_eye_input)
            right_eye_status = "Open" if np.argmax(right_eye_pred) == 1 else "Closed"

        # Draw eye bounding boxes and labels (visualization)
        cv2.rectangle(frame, (left_rect[0], left_rect[1]), 
                      (left_rect[0] + left_rect[2], left_rect[1] + left_rect[3]), 
                      (0, 255, 255), 2)  # Yellow box for left eye
        
        cv2.rectangle(frame, (right_rect[0], right_rect[1]), 
                      (right_rect[0] + right_rect[2], right_rect[1] + right_rect[3]), 
                      (0, 255, 255), 2)  # Yellow box for right eye
        
        # Label eye states
        cv2.putText(frame, f"Left: {left_eye_status}", (left_rect[0], left_rect[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if left_eye_status == "Open" else (0, 0, 255), 2)
        
        cv2.putText(frame, f"Right: {right_eye_status}", (right_rect[0], right_rect[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if right_eye_status == "Open" else (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()