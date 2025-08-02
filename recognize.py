import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from data_preparation import load_processed_data
import os

# Import your custom layer from model.py
from model import ArcFace

class FaceRecognizer:
    def __init__(self, classification_model_path, siamese_model_path, threshold=0.5):
        # Load models with custom objects
        self.classification_model = tf.keras.models.load_model(
            classification_model_path,
            custom_objects={'ArcFace': ArcFace}
        )
        self.siamese_model = tf.keras.models.load_model(
            siamese_model_path,
            custom_objects={'ArcFace': ArcFace}
        )
        
        # Load label dictionary
        self.label_dict = {}
        with open('processed_data/label_dict.txt', 'r') as f:
            for line in f:
                label, name = line.strip().split(':')
                self.label_dict[int(label)] = name
        
        # Threshold for verification
        self.threshold = threshold
        
        # Face detection model (Haar cascade as fallback)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def preprocess_image(self, img):
        """Preprocess image for model input"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        return np.expand_dims(img, axis=0)
    
    def detect_faces(self, frame):
        """Detect faces in a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))
        return faces
    
    def recognize_face(self, img):
        """Recognize a face using classification model"""
        processed_img = self.preprocess_image(img)
        predictions = self.classification_model.predict(processed_img)
        predicted_label = np.argmax(predictions)
        confidence = np.max(tf.nn.softmax(predictions))
        name = self.label_dict.get(predicted_label, "Unknown")
        return name, confidence
    
    def verify_face(self, img1, img2):
        """Verify if two faces belong to the same person"""
        img1_processed = self.preprocess_image(img1)
        img2_processed = self.preprocess_image(img2)
        
        similarity = self.siamese_model.predict([img1_processed, img2_processed])
        return similarity[0][0] > self.threshold, similarity[0][0]
    
    def recognize_from_camera(self):
        """Real-time face recognition from camera"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_img = frame[y:y+h, x:x+w]
                
                # Recognize face
                name, confidence = self.recognize_face(face_img)
                
                # Draw rectangle and label
                color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                label = f"{name} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Display result
            cv2.imshow('Face Recognition', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def debug_predictions(self, img):
        processed = self.preprocess_image(img)
        predictions = self.classification_model.predict(processed)
        print("Raw predictions:", predictions)
        print("Softmax probabilities:", tf.nn.softmax(predictions).numpy())
        return np.argmax(predictions)
    def test_with_images(self, test_dir ='data'):
        for person in os.listdir(test_dir):
            person_dir = os.path.join(test_dir, person)
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                
                # Detect and recognize
                faces = self.detect_faces(img)
                for (x,y,w,h) in faces:
                    face_img = img[y:y+h, x:x+w]
                    name, confidence = self.recognize_face(face_img)
                    
                    # Display results
                    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(img, f"{name} ({confidence:.2f})", 
                            (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                            (0,255,0), 2)
                
                cv2.imshow('Test Result', img)
                cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == '__main__':
    # Initialize recognizer
    recognizer = FaceRecognizer(
        classification_model_path='models/classification_model.keras',
        siamese_model_path='models/siamese_model.keras',
        threshold=0.7
    )
    
    # Run real-time recognition
    # print("Starting face recognition from camera...")
    # print("Press 'q' to quit.")
    recognizer.recognize_from_camera()
    # Test with images in a directory
    # recognizer.test_with_images('data')