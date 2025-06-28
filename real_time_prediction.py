# real_time_prediction.py
# Real-time Body Language Prediction using trained model

import cv2
import numpy as np
import pickle
import mediapipe as mp
from collections import deque

class BodyLanguagePredictor:
    def __init__(self, model_path='Models/body_language_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.model_info = None
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Smoothing for predictions
        self.prediction_buffer = deque(maxlen=10)
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open('Models/model_info.pkl', 'rb') as f:
                self.model_info = pickle.load(f)
            
            print(f"Model loaded: {self.model_info['model_name']}")
            print(f"Classes: {self.model_info['classes']}")
            print(f"Test Accuracy: {self.model_info['test_accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results"""
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
    
    def smooth_prediction(self, prediction, confidence):
        """Smooth predictions using a buffer"""
        self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < 3:
            return prediction, confidence
        
        # Get most frequent prediction in buffer
        predictions = [p[0] for p in self.prediction_buffer]
        confidences = [p[1] for p in self.prediction_buffer]
        
        # Find most common prediction
        unique_predictions = list(set(predictions))
        prediction_counts = [predictions.count(p) for p in unique_predictions]
        most_common_idx = prediction_counts.index(max(prediction_counts))
        
        smoothed_prediction = unique_predictions[most_common_idx]
        avg_confidence = np.mean([c for p, c in self.prediction_buffer if p == smoothed_prediction])
        
        return smoothed_prediction, avg_confidence
    
    def run_prediction(self):
        """Run real-time body language prediction"""
        print("Starting real-time body language prediction...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        ) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Make detections
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                
                # Draw the annotations on the image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                    self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )
                
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                )
                
                self.mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                )
                
                self.mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
                # Make prediction
                try:
                    keypoints = self.extract_keypoints(results)
                    keypoints = np.nan_to_num(keypoints)
                    
                    # Make prediction
                    X = keypoints.reshape(1, -1)
                    prediction = self.model.predict(X)[0]
                    confidence = np.max(self.model.predict_proba(X))
                    
                    # Smooth prediction
                    smoothed_prediction, smoothed_confidence = self.smooth_prediction(prediction, confidence)
                    
                    # Display prediction
                    cv2.rectangle(image, (0, 0), (400, 80), (245, 117, 16), -1)
                    
                    cv2.putText(image, 'BODY LANGUAGE', (15, 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    cv2.putText(image, f'{smoothed_prediction}', (15, 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    cv2.putText(image, f'{smoothed_confidence:.2f}', (15, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Color coding for different emotions
                    color_map = {
                        'Happy': (0, 255, 0),      # Green
                        'Sad': (255, 0, 0),        # Blue
                        'Victorious': (0, 165, 255), # Orange
                        'Fight': (0, 0, 255)       # Red
                    }
                    
                    if smoothed_prediction in color_map:
                        cv2.rectangle(image, (400, 0), (640, 80), color_map[smoothed_prediction], -1)
                        cv2.putText(image, smoothed_prediction, (410, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                except Exception as e:
                    cv2.putText(image, 'Processing...', (15, 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                cv2.imshow('Body Language Detection', image)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run real-time prediction"""
    try:
        predictor = BodyLanguagePredictor()
        predictor.run_prediction()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have trained and saved the model first!")

if __name__ == "__main__":
    main()