import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from twilio.rest import Client
import requests
from datetime import datetime
from collections import deque
import gdown


import shutil

# Ensure .streamlit folder exists
os.makedirs(".streamlit", exist_ok=True)

# Copy secret file from Render secret location to Streamlit expected location
source_secret_path = "/etc/secrets/.secrets.toml"
dest_secret_path = ".streamlit/secrets.toml"

if os.path.exists(source_secret_path):
    shutil.copy(source_secret_path, dest_secret_path)

# Load secrets
TWILIO_ACCOUNT_SID = st.secrets["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
TWILIO_SENDER_NUMBER = st.secrets["TWILIO_SENDER_NUMBER"]
TWILIO_RECIPIENT_NUMBER = st.secrets["TWILIO_RECIPIENT_NUMBER"]
GDRIVE_MODEL_ID = st.secrets["MODEL_DRIVE_ID"]

MODEL_PATH = "movileV3_89.keras"
IMG_SIZE = (224, 224)
FRAME_SKIP = 3
PREDICTION_HISTORY_LENGTH = 100  # Number of predictions to keep for visualization
VIOLENCE_THRESHOLD = 0.5
SMS_ALERT_THRESHOLD = 20  # Number of violent frames to trigger SMS

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

st.set_page_config(layout="wide")
st.title("🚨 Compact Violence Detection")
st.write("Upload a video to analyze frame-by-frame violence probability with compact display.")


def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model securely from Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}", MODEL_PATH, quiet=False)

@st.cache_resource
def load_violence_model():
    download_model_if_needed()
    try:
        model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()

        
def get_device_location():
    """Get approximate location using IP address with better error handling"""
    try:
        response = requests.get('https://ipinfo.io/json', timeout=3)
        if response.status_code == 200:
            data = response.json()
            city = data.get('city', 'Unknown city')
            country = data.get('country', 'Unknown country')
            return f"{city}, {country}"
        
        response = requests.get('https://geolocation-db.com/json/', timeout=3)
        if response.status_code == 200:
            data = response.json()
            city = data.get('city', 'Unknown city')
            country = data.get('country_name', 'Unknown country')
            return f"{city}, {country}"
        
        return "Location detection failed"
        
    except requests.exceptions.RequestException:
        return "Could not connect to location services"
    except Exception as e:
        print(f"Location error: {str(e)}")
        return "Location unavailable"

def send_sms_alert(violent_count):
    """Send SMS alert with location and timestamp"""
    location = get_device_location()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        message = twilio_client.messages.create(
            body=f"🚨 VIOLENCE ALERT\n"
                 f"Around Location: {location}\n"
                 f"Time: {timestamp}\n"
                 f"Violent frames detected: {violent_count}",
            from_=TWILIO_SENDER_NUMBER,
            to=TWILIO_RECIPIENT_NUMBER
        )
        return message.sid
    except Exception as e:
        st.error(f"Failed to send SMS: {str(e)}")
        return None
    

    
def preprocess_frame(frame):
    """Prepare frame for model prediction"""
    frame = cv2.resize(frame, IMG_SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return img_to_array(frame) / 255.0

def add_prediction_overlay(frame, pred):
    """Add prediction text overlay to the frame"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Color based on prediction (red for violent, green for safe)
    color = (0, 0, 255) if pred > VIOLENCE_THRESHOLD else (0, 255, 0)
    cv2.putText(frame, f"Violence: {pred:.2f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

def analyze_webcam(model):
    """Analyze live webcam feed"""
    st.subheader("Live Webcam Analysis Can Only Be Tested Locally So Contact From GitHub")
    
    # Create layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("Webcam Feed")
        webcam_placeholder = st.empty()
    
    with col2:
        st.write("Probability Timeline")
        chart_placeholder = st.empty()
    
    # Initialize chart
    fig, ax = plt.subplots(figsize=(10, 3))
    predictions = deque(maxlen=PREDICTION_HISTORY_LENGTH)
    timestamps = deque(maxlen=PREDICTION_HISTORY_LENGTH)
    line, = ax.plot([], [], 'b-')
    ax.axhline(y=VIOLENCE_THRESHOLD, color='r', linestyle='--')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability')
    
    # Status area
    status_area = st.empty()
    alert_placeholder = st.empty()
    stop_button = st.button("Stop Webcam Analysis")
    
    # Initialize counters
    frame_count = 0
    violent_count = 0
    sms_sent = False
    start_time = time.time()
    
    # Webcam capture loop
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam!")
        return
    
    try:
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not read frame from webcam")
                break
                
            # Process every FRAME_SKIP-th frame
            if frame_count % FRAME_SKIP == 0:
                # Preprocess and predict
                processed_frame = preprocess_frame(frame)
                pred = model.predict(np.expand_dims(processed_frame, axis=0), verbose=0)[0][0]
                current_time = time.time() - start_time
                
                # Store results
                predictions.append(pred)
                timestamps.append(current_time)
                
                # Count violent frames
                if pred > VIOLENCE_THRESHOLD:
                    violent_count += 1
                
                # Send SMS alert if threshold reached and not already sent
                if violent_count >= SMS_ALERT_THRESHOLD and not sms_sent:
                    sms_id = send_sms_alert(violent_count)
                    if sms_id:
                        alert_placeholder.success(f"📱 SMS Alert Sent! (ID: {sms_id})")
                        sms_sent = True
                
                # Display frame with overlay
                display_frame = cv2.resize(frame, (400, 300))
                display_frame = add_prediction_overlay(display_frame, pred)
                webcam_placeholder.image(display_frame, channels="BGR")

                
                # Update status
                status_area.text(f"Frames processed: {frame_count} | "
                               f"Violent frames: {violent_count} | "
                               f"Elapsed: {current_time:.1f}s")
                
                # Update chart periodically
                if frame_count % (FRAME_SKIP*5) == 0 and len(predictions) > 1:
                    line.set_data(timestamps, predictions)
                    ax.set_xlim(max(0, current_time-10), max(10, current_time))
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)
            
            frame_count += 1
            
            # Add a small delay to prevent high CPU usage
            time.sleep(0.01)
            
            # Check if stop button was pressed
            if stop_button:
                break
                
    except Exception as e:
        st.error(f"Error during webcam analysis: {str(e)}")
    finally:
        cap.release()
        st.info("Webcam released")
        
        # Show final stats
        st.subheader("Webcam Session Summary")
        st.write(f"Total frames processed: {frame_count}")
        st.write(f"Violent frames detected: {violent_count}")
        if violent_count >= SMS_ALERT_THRESHOLD:
            st.success(f"SMS alert was sent from {get_device_location()}")

def analyze_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video file!")
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create compact display layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Live Analysis")
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        alert_placeholder = st.empty()
    
    with col2:
        st.subheader("Probability Timeline")
        chart_placeholder = st.empty()
    
    # Initialize chart
    fig, ax = plt.subplots(figsize=(10, 3))
    predictions = []
    timestamps = []
    line, = ax.plot([], [], 'b-')
    ax.axhline(y=VIOLENCE_THRESHOLD, color='r', linestyle='--')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability')
    
    frame_count = 0
    start_time = time.time()
    violent_count = 0
    sms_sent = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % FRAME_SKIP == 0:
            # Process and predict
            processed_frame = preprocess_frame(frame)
            pred = model.predict(np.expand_dims(processed_frame, axis=0), verbose=0)[0][0]
            timestamp = frame_count / fps
            
            # Add results
            predictions.append(pred)
            timestamps.append(timestamp)
            
            # Count violent frames
            if pred > VIOLENCE_THRESHOLD:
                violent_count += 1
            
            # Send SMS alert if threshold reached and not already sent
            if violent_count >= SMS_ALERT_THRESHOLD and not sms_sent:
                sms_id = send_sms_alert(violent_count)
                if sms_id:
                    alert_placeholder.success(f"📱 SMS Alert Sent! (ID: {sms_id})")
                    sms_sent = True
            
            # Display compact video with overlay
            display_frame = cv2.resize(frame, (400, 300))
            display_frame = add_prediction_overlay(display_frame, pred)
            video_placeholder.image(display_frame, channels="BGR")

            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processed: {frame_count}/{total_frames} frames | "
                           f"Violent: {violent_count} | "
                           f"Elapsed: {time.time()-start_time:.1f}s")
            
            # Update chart every 5 frames
            if frame_count % (FRAME_SKIP*5) == 0:
                line.set_data(timestamps, predictions)
                ax.set_xlim(0, max(10, max(timestamps)+1))
                chart_placeholder.pyplot(fig)
                plt.close(fig)
            
        frame_count += 1
    
    cap.release()
    return {
        'predictions': predictions,
        'timestamps': timestamps,
        'fps': fps,
        'total_frames': total_frames,
        'violent_count': violent_count,
        'location': get_device_location()
    }

def show_final_results(results):
    st.subheader("Analysis Results")
    
    # Final chart
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(results['timestamps'], results['predictions'], 'b-')
    ax.axhline(y=VIOLENCE_THRESHOLD, color='r', linestyle='--')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Probability')
    st.pyplot(fig)
    
    # Violent segments
    threshold = st.slider("Violence Threshold", 0.0, 1.0, VIOLENCE_THRESHOLD, 0.01)
    violent_frames = [i for i, p in enumerate(results['predictions']) if p > threshold]
    
    if violent_frames:
        st.write(f"🔴 Detected {len(violent_frames)} violent frames (>{threshold})")
        violent_times = [results['timestamps'][i] for i in violent_frames]
        st.write("Violent moments (seconds):", ", ".join(f"{t:.1f}" for t in violent_times[:10]))
    
    # Display total violent count
    st.warning(f"Total violent frames detected: {results['violent_count']}")
    if results['violent_count'] >= SMS_ALERT_THRESHOLD:
        st.success(f"SMS alert was sent from {results['location']}")

def main():
    model = load_violence_model()
    
    # Add mode selection
    analysis_mode = st.radio(
        "Select analysis mode:",
        ("Upload Video", "Use Webcam"),
        horizontal=True
    )
    
    if analysis_mode == "Upload Video":
        uploaded_file = st.file_uploader("Choose video file", type=["mp4", "avi", "mov"])
        
        if uploaded_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(uploaded_file.read())
                    temp_path = tmp.name
                
                results = analyze_video(temp_path, model)
                
                if results:
                    show_final_results(results)
            
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
            
            finally:
                try:
                    if 'temp_path' in locals():
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                except Exception as e:
                    st.warning(f"Could not delete temporary file: {str(e)}")
    
    elif analysis_mode == "Use Webcam":
        st.info("Click the button below to start webcam analysis")
        if st.button("Start Webcam Analysis"):
            analyze_webcam(model)

if __name__ == "__main__":
    main()
