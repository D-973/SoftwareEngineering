import cv2
import numpy as np
import os
import sys

def check_dependencies():
    """Check if all required dependencies and files are available"""
    try:
        # Check OpenCV version
        print(f"OpenCV Version: {cv2.__version__}")
        
        # Check if camera is available
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot access camera")
            return False
        cap.release()
        
        return True
        
    except Exception as e:
        print(f"Dependency check failed: {str(e)}")
        return False

class SimpleLivenessDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_liveness(self, frame, face_location):
        """Simple liveness detection based on eye detection and basic texture analysis"""
        try:
            x, y, w, h = face_location
            face_roi = frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(
                gray_roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Basic texture analysis
            variance = np.var(gray_roi)
            
            # Combined check
            has_eyes = len(eyes) >= 2
            has_texture = variance > 500  # Threshold may need adjustment
            
            return has_eyes and has_texture
            
        except Exception as e:
            print(f"Error in liveness detection: {str(e)}")
            return False

def main():
    try:
        # Check dependencies first
        if not check_dependencies():
            print("Failed dependency check. Exiting...")
            return

        # Initialize recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Get current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to load the trainer file
        trainer_path = os.path.join(current_dir, 'trainer', 'trainer.yml')
        try:
            recognizer.read(trainer_path)
        except Exception as e:
            print(f"Error loading trainer file from {trainer_path}")
            print(f"Error details: {str(e)}")
            return

        # Initialize face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error: Could not load face cascade classifier")
            return

        # Initialize liveness detector
        liveness_detector = SimpleLivenessDetector()

        # Initialize camera
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Cannot access camera")
            return

        # Configure camera
        cam.set(3, 640)  # width
        cam.set(4, 480)  # height

        # Names list - customize this
        names = ['None', 'Dhea', 'Atha', 'Ricky']

        print("Starting face recognition. Press 'ESC' to exit.")

        while True:
            ret, img = cam.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(cam.get(3)*0.1), int(cam.get(4)*0.1))
            )

            for (x, y, w, h) in faces:
                # Check liveness
                is_live = liveness_detector.detect_liveness(img, (x, y, w, h))

                if is_live:
                    # Draw green rectangle for real face
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

                    # Recognize face
                    try:
                        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                        
                        if confidence < 100:  # Adjust this threshold as needed
                            id = names[id] if id < len(names) else "unknown"
                            confidence = f"{round(100 - confidence)}%"
                        else:
                            id = "unknown"
                            confidence = f"{round(100 - confidence)}%"

                        # Display results
                        cv2.putText(img, str(id), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        cv2.putText(img, str(confidence), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)
                    except Exception as e:
                        print(f"Error during face recognition: {str(e)}")
                else:
                    # Draw red rectangle for fake face
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
                    cv2.putText(img, "Fake Face", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            cv2.imshow('Face Recognition', img)

            # Break loop with 'ESC' key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Cleanup
        print("Closing application...")
        cam.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Critical error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()