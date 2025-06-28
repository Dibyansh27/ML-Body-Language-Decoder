# data_collection.py
# Body Language Data Collection using MediaPipe

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import os
from sklearn.model_selection import train_test_split

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def collect_data(class_name, duration=15):
    """Collect body language data for a specific class"""
    print(f"Starting data collection for {class_name} class...")
    print("Position yourself and press 's' to start recording, 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    data = []
    labels = []
    recording = False
    frame_count = 0
    max_frames = duration * 30  # Assuming 30 FPS
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Make detections
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            
            # Draw landmarks
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            # Draw pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            # Draw left hand landmarks
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # Draw right hand landmarks
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Show recording status
            if recording:
                cv2.putText(image, f'RECORDING {class_name}: {frame_count}/{max_frames}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Extract keypoints and save
                keypoints = extract_keypoints(results)
                data.append(keypoints)
                labels.append(class_name)
                frame_count += 1
                
                if frame_count >= max_frames:
                    print(f"Data collection complete for {class_name}!")
                    break
            else:
                cv2.putText(image, f'Press "s" to start recording {class_name}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('MediaPipe Feed', image)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not recording:
                recording = True
                print(f"Recording {class_name}...")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return data, labels

def main():
    """Main function to collect data for all classes"""
    # Create dataset directory
    os.makedirs('Dataset', exist_ok=True)
    
    all_data = []
    all_labels = []
    
    classes = ['Happy', 'Sad', 'Victorious', 'Fight']
    
    for class_name in classes:
        print(f"\n=== Collecting data for {class_name} ===")
        if class_name == 'Happy':
            print("Show happy facial expressions and gestures")
        elif class_name == 'Sad':
            print("Show sad facial expressions and posture")
        elif class_name == 'Victorious':
            print("Put your hands up with fists (victory pose)")
        elif class_name == 'Fight':
            print("Put your fists in front of your face (fighting stance)")
        
        data, labels = collect_data(class_name)
        all_data.extend(data)
        all_labels.extend(labels)
    
    # Create DataFrame and save
    if all_data:
        df = pd.DataFrame(all_data)
        df['class'] = all_labels
        df.to_csv('Dataset/body_language_data.csv', index=False)
        print(f"\nDataset saved with {len(df)} samples!")
        print(f"Classes distribution:")
        print(df['class'].value_counts())
    else:
        print("No data collected!")

if __name__ == "__main__":
    main()