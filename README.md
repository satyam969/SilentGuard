# 🚨 Real-Time Violence Detection with Streamlit and Twilio

A real-time and video-upload-based violence detection system powered by deep learning. This Streamlit app analyzes video frames (live or uploaded) and sends **instant SMS alerts** using the **Twilio API** when violent activity is detected. It also fetches **approximate location data** using the user's IP address.

---

## 🎯 Features

- 🎥 **Real-Time Webcam Support** 
  Run the app with webcam access to detect violent behavior live.

- 📁 **Video Upload Mode**  
  Upload any `.mp4`, `.avi`, or `.mov` file and get a frame-by-frame analysis.

- 📊 **Live Probability Timeline**  
  Visual chart displaying frame-wise violence probability with threshold indicator.

- 🧠 **Deep Learning-Based Detection**  
  Uses a fine-tuned `.keras` model (e.g., MobileNetV3) for binary classification.

- 📱 **SMS Alerting with Twilio**  
  Sends an alert SMS with location + timestamp when violent activity exceeds a certain frame threshold.

- 🌍 **IP-Based Geolocation**  
  Retrieves the user's approximate city and country during alert dispatch.

---

## 🖼️ App Layout

- 📺 **Left Panel** – Compact video feed with overlay showing violence probability
- 📈 **Right Panel** – Timeline plot showing how prediction evolves over time
- 📊 **Summary Section** – Highlights timestamps of detected violence and shows alert status

---

## 🚀 Getting Started Locally

### 📦 Requirements

- Python 3.10 or 3.11
- Streamlit
- TensorFlow
- OpenCV
- Twilio
- Matplotlib

### 🔧 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/real-time-violence-alert.git
cd real-time-violence-alert
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Place Your Model**

Add your trained `.keras` file (e.g., `movileV3_89.keras`) to the root directory.

4. **Add Twilio Secrets**

Create a `.streamlit/secrets.toml` file with the following content:

```toml
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_AUTH_TOKEN = "your_auth_token"
TWILIO_SENDER_NUMBER = "+1xxxxxxxxxx"
TWILIO_RECIPIENT_NUMBER = "+91xxxxxxxxxx"
```

---

## 🧠 Model Overview

- Format: `.keras`
- Input Size: 224x224x3
- Task: Binary classification (Violence / No Violence)
- Backend: TensorFlow / Keras

---

## 🖥️ Modes of Operation

### 1. 📼 **Video Upload Mode**
- Upload a file from your local device
- App processes video frame-by-frame
- Shows live overlay + sends SMS if violence exceeds threshold

### 2. 🎥 **Live Camera Mode (optional)**
- Modify the code slightly to capture from webcam (`cv2.VideoCapture(0)`)
- Analyze violence in real-time as video is streamed from your webcam

> This is ideal for **CCTV monitoring** or **live security feeds**.

---

## ☁️ Deploy on Streamlit Cloud

1. Push the project to a GitHub repo
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Link your GitHub, deploy the repo
4. In **Settings → Secrets**, paste the same Twilio credentials

---

## 📱 Sample SMS Output

```
🚨 VIOLENCE ALERT
Approximate Location: Mumbai, IN
Time: 2025-06-30 19:45:10
Violent frames detected: 27
```

---

## 📸 Screenshots / Demo

> *(Add screenshots or a short demo video/GIF here to visually explain the UI)*

---

## 🧑‍💻 Author

**Satyam Tiwari**  
🎓 NIT Jamshedpur • 📫 [LinkedIn](https://www.linkedin.com/in/priya-raj-4b0380273) • 🌐 [Portfolio](https://my-portfolio-one-lilac-88.vercel.app)

---

## ⚠️ Disclaimer

- This project is for research and educational purposes.
- Real-world deployment should comply with **privacy laws** and **surveillance regulations**.
- Model performance depends on training data and may require improvements for high-stakes usage.

---

## 📜 License

MIT License – Feel free to use and modify this project with credit.
