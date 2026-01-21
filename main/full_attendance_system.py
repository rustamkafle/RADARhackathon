import cv2
import numpy as np
import joblib
import requests
import time
import pandas as pd
from skimage.feature import hog
from datetime import datetime
from openpyxl import load_workbook
import os

# Configuration
ESP32_DISPLAY_IP = "http://172.20.10.11"
MODEL_PATH = "../create_model/data/face_svm_model.pkl"
EXCEL_DIR = "../view_the_attendance"  # ✅ Path to Excel files

# Load the trained SVM model
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")

def preprocess_face(frame):
    """Preprocess face for HOG features."""
    resized = cv2.resize(frame, (50, 50))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return hog_feat.reshape(1, -1)

def send_message_to_esp32(message):
    """Send message to ESP32 display."""
    try:
        response = requests.post(
            f"{ESP32_DISPLAY_IP}/displayinfo",
            json={"message": message},
            headers={"Content-Type": "application/json"},
            timeout=2
        )
        if response.status_code == 200:
            print(f"✅ Message sent: {message}")
        else:
            print(f"❌ Error: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def get_button_info():
    """Fetch button info from ESP32."""
    try:
        response = requests.get(f"{ESP32_DISPLAY_IP}/buttoninfo", timeout=2)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error fetching button info: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error fetching button info: {str(e)}")
        return None



def update_attendance_sheet(file_path, student_name):
    """Update the attendance sheet with the student roll number and current time."""
    print(f"\n{'='*60}")
    print(f"📝 UPDATING ATTENDANCE SHEET")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    print(f"Roll Number: {student_name}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Full path: {os.path.abspath(file_path)}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File '{file_path}' does not exist!")
        print(f"   Please create {file_path} with at least 'Roll Number' and 'Name of Students' columns")
        return

    today = datetime.now().strftime("%b %d, %Y")
    current_time = datetime.now().strftime("%I:%M %p")
    timestamp_col = f"Timestamp for {today.split(',')[0]}"  # "Timestamp for Jan 10"

    print(f"Date: {today}")
    print(f"Time: {current_time}")
    print(f"Timestamp Column: {timestamp_col}")

    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        print(f"✅ Successfully read Excel file")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"   Column names: {list(df.columns)}")

        # Verify required columns exist
        if "Roll Number" not in df.columns:
            print(f"❌ ERROR: 'Roll Number' column not found!")
            return

        # Show first few roll numbers
        print(f"   Roll numbers in sheet: {df['Roll Number'].dropna().tolist()[:5]}...")

    except Exception as e:
        print(f"❌ ERROR reading Excel file: {str(e)}")
        return

    # ✅ Add today's date column if it doesn't exist
    if today not in df.columns:
        print(f"➕ Adding new attendance column: '{today}'")
        # Insert after "Roll Number" column
        roll_col_idx = df.columns.get_loc("Roll Number")
        df.insert(roll_col_idx + 1, today, "")
    else:
        print(f"✅ Column '{today}' already exists")

    # ✅ Add timestamp column if it doesn't exist
    if timestamp_col not in df.columns:
        print(f"➕ Adding new timestamp column: '{timestamp_col}'")
        # Insert right after today's date column
        today_col_idx = df.columns.get_loc(today)
        df.insert(today_col_idx + 1, timestamp_col, "")
    else:
        print(f"✅ Column '{timestamp_col}' already exists")

    # Find the row with the student's roll number
    print(f"\n🔍 Searching for roll number: '{student_name}'")

    student_row = df[df["Roll Number"] == student_name]  # prediction is roll number

    if student_row.empty:
        print(f"❌ Roll Number '{student_name}' NOT FOUND in the sheet!")
        print(f"\n   Available roll numbers (first 10):")
        for i, rn in enumerate(df["Roll Number"].dropna().head(10)):
            print(f"      {i+1}. {rn}")
        print(f"\n   💡 TIP: Make sure the roll number matches exactly")
        return

    index = student_row.index[0]
    roll_number = df.at[index, "Roll Number"]
    full_name = df.at[index, "Name of Students"]

    print(f"✅ Found student at row index {index}")
    print(f"   Roll Number: {roll_number}")
    print(f"   Full name in sheet: {full_name}")

    # Check if already marked present
    current_status = df.at[index, today]
    if current_status == "P":
        print(f"⚠️  Student already marked present today!")
        print(f"   Previous timestamp: {df.at[index, timestamp_col]}")
        print(f"   Skipping duplicate entry...")
        print(f"{'='*60}\n")
        return

    # Update attendance
    df.at[index, today] = "P"
    df.at[index, timestamp_col] = current_time

    # Save to Excel
    try:
        df.to_excel(file_path, index=False)
        print(f"✅✅✅ SUCCESS! Marked {full_name} present")
        print(f"   Roll Number: {roll_number}")
        print(f"   Status: P")
        print(f"   Time: {current_time}")
    except Exception as e:
        print(f"❌ ERROR saving Excel file: {str(e)}")
        print(f"   Make sure the file is not open in Excel!")

    print(f"{'='*60}\n")


def mark_absent(file_path):
    """Mark all students as absent who haven't been marked present."""
    print(f"\n{'='*60}")
    print(f"📋 FINALIZING ATTENDANCE - Marking Absent Students")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    
    today = datetime.now().strftime("%b %d, %Y")
    timestamp_col = f"Timestamp for {today.split(',')[0]}"

    if not os.path.exists(file_path):
        print(f"❌ File {file_path} not found.")
        print(f"{'='*60}\n")
        return

    try:
        df = pd.read_excel(file_path)
        print(f"✅ Successfully read Excel file")
        print(f"   Total students: {len(df)}")
    except Exception as e:
        print(f"❌ ERROR reading Excel file: {str(e)}")
        print(f"{'='*60}\n")
        return

    # Check if today's column exists
    if today not in df.columns:
        print(f"❌ ERROR: Today's date column '{today}' not found!")
        print(f"   This means no one was marked present today.")
        print(f"   Available columns: {list(df.columns)}")
        print(f"\n   Creating the column and marking all as absent...")
        
        # Create the columns
        name_col_idx = df.columns.get_loc("Name of Students")
        df.insert(name_col_idx + 1, today, "A")  # Mark all absent
        df.insert(name_col_idx + 2, timestamp_col, "")
        
        try:
            df.to_excel(file_path, index=False)
            print(f"✅ Marked all {len(df)} students as absent")
        except Exception as e:
            print(f"❌ ERROR saving Excel file: {str(e)}")
        
        print(f"{'='*60}\n")
        return

    # Count current attendance
    present_before = (df[today] == "P").sum()
    absent_before = (df[today] == "A").sum()
    unmarked = len(df) - present_before - absent_before
    
    print(f"   Currently Present: {present_before}")
    print(f"   Currently Absent: {absent_before}")
    print(f"   Unmarked: {unmarked}")

    # Mark all unmarked students as absent
    df[today] = df[today].apply(lambda x: x if x == "P" else "A")
    
    # Save the file
    try:
        df.to_excel(file_path, index=False)
        
        # Recalculate final stats
        present_final = (df[today] == "P").sum()
        absent_final = (df[today] == "A").sum()
        
        print(f"\n✅ Attendance finalized!")
        print(f"   📊 Final Statistics:")
        print(f"      Present: {present_final} ({present_final/len(df)*100:.1f}%)")
        print(f"      Absent:  {absent_final} ({absent_final/len(df)*100:.1f}%)")
        print(f"      Total:   {len(df)}")
        
        if unmarked > 0:
            print(f"\n   Marked {unmarked} additional students as absent")
            
    except Exception as e:
        print(f"❌ ERROR saving Excel file: {str(e)}")
        print(f"   Make sure the file is not open in Excel!")
    
    print(f"{'='*60}\n")

def main():
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("🎥 Starting face recognition from laptop webcam...")
    print(f"📺 Display URL: {ESP32_DISPLAY_IP}/displayinfo")
    print(f"📂 Working directory: {os.getcwd()}")
    print("Press 'q' to quit")
    print("-" * 50)
    
    # ✅ FIX: Check if Excel files exist, if not check parent directory
    attendance_dir = "../view_the_attendance"
    if not os.path.exists("subject1.xlsx") and os.path.exists(f"{attendance_dir}/subject1.xlsx"):
        print(f"📁 Excel files found in {attendance_dir}/")
        os.chdir(attendance_dir)
        print(f"📂 Changed working directory to: {os.getcwd()}")
    
    # Verify files exist
    if not os.path.exists("subject1.xlsx"):
        print("❌ ERROR: subject1.xlsx not found!")
        print(f"   Please create subject1.xlsx in {os.getcwd()}")
    else:
        print("✅ subject1.xlsx found")
        
    if not os.path.exists("subject2.xlsx"):
        print("❌ ERROR: subject2.xlsx not found!")
        print(f"   Please create subject2.xlsx in {os.getcwd()}")
    else:
        print("✅ subject2.xlsx found")
    
    print("-" * 50)

    last_prediction = None
    detection_start_time = None
    current_subject = None
    start_flag = False
    marked_students = set()
    last_button_check = 0

    while True:
        current_time = time.time()
        
        # Check button info every 0.5 seconds
        if current_time - last_button_check >= 0.5:
            button_info = get_button_info()
            if button_info:
                # Handle SUBJECT selection (event-based)
                if button_info["subject1"] == 1:
                    print("\n" + "="*60)
                    print("📚 SUBJECT 1 SELECTED")
                    print("="*60)
                    current_subject = os.path.join(EXCEL_DIR, "subject1.xlsx")  # ✅ Use full path
                    print(f"Target file: {current_subject}")
                    if os.path.exists(current_subject):
                        print(f"✅ File exists")
                    else:
                        print(f"❌ WARNING: File does not exist!")
                    print("="*60 + "\n")
                    
                elif button_info["subject2"] == 1:
                    print("\n" + "="*60)
                    print("📚 SUBJECT 2 SELECTED")
                    print("="*60)
                    current_subject = os.path.join(EXCEL_DIR, "subject2.xlsx")  # ✅ Use full path
                    print(f"Target file: {current_subject}")
                    if os.path.exists(current_subject):
                        print(f"✅ File exists")
                    else:
                        print(f"❌ WARNING: File does not exist!")
                    print("="*60 + "\n")
                
                # Handle START button (event-based)
                if button_info["start"] == 1:
                    print("\n" + "="*60)
                    print("▶️ START - Attendance tracking ACTIVE")
                    print(f"Subject file: {current_subject}")
                    print("="*60 + "\n")
                    start_flag = True
                    marked_students.clear()
                    detection_start_time = None
                    last_prediction = None
                
                # Handle STOP button (event-based)
                if button_info["stop"] == 1:
                    print("\n" + "="*60)
                    print("⏹ STOP - Finalizing attendance")
                    print("="*60)
                    if start_flag and current_subject:
                        mark_absent(current_subject)
                    start_flag = False
                    current_subject = None
                    marked_students.clear()
                    detection_start_time = None
                    last_prediction = None
                    print("="*60 + "\n")
                    
            last_button_check = current_time

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame")
            time.sleep(0.1)
            continue

        # Create a copy for display
        display_frame = frame.copy()

        # Block recognition when system is STOPPED
        if not start_flag or not current_subject:
            cv2.putText(display_frame, "SYSTEM STOPPED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display_frame, "Press Subject + Start on ESP32", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Attendance System', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # System is ACTIVE - proceed with face recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        prediction_text = "No face detected"
        confidence_text = ""
        status_text = "Waiting..."
        
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            face_roi = frame[y:y+h, x:x+w]
            hog_features = preprocess_face(face_roi)
            prediction = model.predict(hog_features)[0]
            probabilities = model.predict_proba(hog_features)[0]
            confidence = np.max(probabilities)

            print(f"👤 Detected: {prediction} (Confidence: {confidence:.2%})")
            
            prediction_text = f"Detected: {prediction}"
            confidence_text = f"Confidence: {confidence:.2%}"

            if confidence > 0.6:
                if prediction == last_prediction:
                    if detection_start_time is None:
                        detection_start_time = current_time
                        print(f"   ⏱️  Starting 2-second timer for {prediction}...")
                    elif current_time - detection_start_time >= 2:
                        # 3 seconds stable - mark attendance
                        if prediction not in marked_students:
                            print(f"\n{'='*60}")
                            print(f"🎯 3 SECONDS STABLE - MARKING ATTENDANCE")
                            print(f"{'='*60}")
                            update_attendance_sheet(current_subject, prediction)
                            marked_students.add(prediction)
                            send_message_to_esp32(f"{prediction}\nMarked Present")
                            status_text = f"{prediction} - MARKED PRESENT"
                            print(f"✅ Added to marked students list (Total: {len(marked_students)})")
                        else:
                            print(f"   ℹ️  {prediction} already marked present - skipping")
                        detection_start_time = None
                    else:
                        time_left = 2 - (current_time - detection_start_time)
                        status_text = f"Stable: {time_left:.1f}s remaining"
                        # Only print every 0.5 seconds to reduce spam
                        if int(time_left * 2) != int((time_left + 0.05) * 2):
                            print(f"   ⏱️  Hold steady... {time_left:.1f}s")
                else:
                    # New person detected
                    if last_prediction is not None:
                        print(f"   🔄 Face changed from {last_prediction} to {prediction}")
                    last_prediction = prediction
                    detection_start_time = current_time
                    send_message_to_esp32(prediction)
                    status_text = "New person detected"
            else:
                if detection_start_time is not None:
                    print(f"   ⚠️  Confidence dropped below 60% - resetting timer")
                detection_start_time = None
                status_text = "Low confidence"
        else:
            if last_prediction is not None:
                send_message_to_esp32("No Face Detected")
                last_prediction = None
            detection_start_time = None

        # Add system status overlay
        system_status = f"ACTIVE - {current_subject}"
        status_color = (0, 255, 0)

        # Add text overlays
        cv2.putText(display_frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"System: {system_status}", (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(display_frame, f"Marked: {len(marked_students)}", (10, display_frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Attendance System', display_frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Quitting...")
            break

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping face recognition...")
        print("Goodbye! 👋")
        cv2.destroyAllWindows()
