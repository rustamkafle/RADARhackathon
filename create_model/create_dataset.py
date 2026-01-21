import os
import cv2
import numpy as np
import pickle

def get_ready(studentName):
    cap = cv2.VideoCapture(2) 
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        _, frame = cap.read()
        cv2.putText(frame, f"Train Images of \"{studentName}\"", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),4)
        cv2.putText(frame, f"Press s to start", (50,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),4)
        cv2.putText(frame, f"Press q to quit", (50,150), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),4)
        cv2.imshow("student face", frame)
        keypressed = cv2.waitKey(1)
        if keypressed==ord('s'):
            cap.release()
            cv2.destroyAllWindows()
            train_my_emotion(studentName)
            break
        if keypressed==ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


def train_my_emotion(studentName):
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    count=0
    features = []
    labels = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to capture")
            break
        frame = cv2.resize(frame, (640, 480))  # Resize for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))  # Adjust minSize

        if len(faces) == 0:
            cv2.imshow("student face", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue
        elif len(faces) == 1:
            x, y, w, h = faces[0]
        else:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        sample = frame[y:y+h, x:x+w] 
        sample_resized = cv2.resize(sample, (50,50))
        cv2.putText(frame, "Press 's' to Save", (50,25), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)
        cv2.putText(frame, f"{count}", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)

        if cv2.waitKey(1)==ord('s'):
            cv2.imwrite(f"students/{studentName}-{count}.jpg", sample_resized)

            features.append(sample_resized.flatten())
            labels.append(studentName)
            count += 1

        cv2.imshow("student face", frame)
        if cv2.waitKey(1)==ord('q') or len(labels)==5000:
            break

    features = np.asarray(features)
    labels = np.array(labels)

    feature_file = "data/features.pkl"
    label_file = "data/labels.pkl"

    if os.path.exists(feature_file):
        with open(feature_file, "rb") as f:
            old_features = pickle.load(f)
        
        if old_features.size > 0:
            features = np.vstack([old_features, features])
    
    if os.path.exists(label_file):
        with open(label_file, "rb") as f:
            old_labels = pickle.load(f)
        
        if old_labels.size > 0:
            labels = np.hstack([old_labels, labels])

    # Save features and labels
    with open(feature_file, "wb") as f:
        pickle.dump(features, f)

    with open(label_file, "wb") as f:
        pickle.dump(labels, f)


if __name__=="__main__":
    # Check and create 'students' directory if not exists
    if not os.path.exists("students"):
        os.makedirs("students")
        
    student_names = ["079BEI001", "079BEI002", "079BEI003", "079BEI004"]
    
    for student in student_names:
        get_ready(student)
        
    print("Thank you for your time!")
