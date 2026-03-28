import sys
import os
import streamlit as st
import gdown
import cv2
import tempfile
import numpy as np
import folium
import pandas as pd
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# =====================================================
# 🔧 PATH + MODEL SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "best.pt")
FILE_ID = "1J391ph0-jzdI8vMh5WwZKgqZCl6wUenh"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =====================================================
# 📥 MODEL LOADER (CACHED)
# =====================================================
@st.cache_resource
def load_detector():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading AI Model... ⏳")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model Ready ✅")

    from utils.detector import PotholeDetector
    return PotholeDetector(MODEL_PATH)

detector = load_detector()

# =====================================================
# 📦 UTIL IMPORTS
# =====================================================
from utils.tracker import SimpleTracker
from utils.report import generate_summary
from utils.map_utils import create_map
from utils.visualization import (
    plot_bar_chart,
    plot_pie_chart,
    show_risk_indicator,
    live_dashboard,
    plot_time_series,
    plot_heatmap
)

# =====================================================
# 🎨 UI CONFIG
# =====================================================
st.set_page_config(layout="wide")

st.markdown("""
<style>
body {background-color: #0E1117;}
h1, h2 {color: #00FFAA;}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Smart Road Damage Detection System")

BASE_LAT = 23.0225
BASE_LON = 72.5714

# =====================================================
# 📂 SECTION 1 — IMAGE / VIDEO UPLOAD
# =====================================================
st.header("📂 Upload Image / Video")

uploaded_file = st.file_uploader(
    "Upload road image or video",
    type=["jpg", "jpeg", "png", "mp4"]
)

# ---------------- IMAGE ----------------
def process_image(uploaded_file):
    file_bytes = uploaded_file.read()
    frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)

    frame, detections, stats, fps = detector.detect(frame)

    st.image(frame, channels="BGR")

    summary, total, risk = generate_summary(detections)

    st.subheader("📊 Image Report")
    st.write(summary)
    st.success(f"Total Defects: {total}")

    show_risk_indicator(risk)
    plot_bar_chart(summary)
    plot_pie_chart(summary)


# ---------------- VIDEO ----------------
def process_video(uploaded_file):

    st.subheader("🎥 Video Processing")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    tracker = SimpleTracker()
    unique_detections = []

    if st.button("▶️ Start Video Analysis"):

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            frame, detections, stats, fps = detector.detect(frame)

            for d in detections:
                if tracker.is_new(d["bbox"]):
                    unique_detections.append(d)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1

        cap.release()

        df = pd.DataFrame(unique_detections)

        summary, total, risk = generate_summary(unique_detections)

        st.subheader("📊 Video Report")
        st.write(summary)
        st.success(f"Unique Defects: {total}")

        show_risk_indicator(risk)
        plot_bar_chart(summary)
        plot_pie_chart(summary)

        if not df.empty:
            plot_time_series(df)


# Run upload handlers
if uploaded_file:
    if uploaded_file.type.startswith("image"):
        process_image(uploaded_file)
    elif uploaded_file.type.startswith("video"):
        process_video(uploaded_file)

# =====================================================
# 🎥 SECTION 2 — LIVE DETECTION (WEBRTC)
# =====================================================
# =====================================================
# 🎥 LIVE DETECTION (FORCED LAPTOP CAMERA)
# =====================================================
st.header("🎥 Live Detection (Laptop Camera Only)")

class LiveProcessor(VideoProcessorBase):

    def __init__(self):
        self.tracker = SimpleTracker()
        self.unique_detections = []
        self.map_points = []
        self.frame_count = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.resize(img, (640, 480))

        img, detections, stats, fps = detector.detect(img)

        lat = BASE_LAT + self.frame_count * 0.00005
        lon = BASE_LON + self.frame_count * 0.00005

        for d in detections:
            if self.tracker.is_new(d["bbox"]):
                self.unique_detections.append(d)

                if d["label"] == "pothole":
                    self.map_points.append((lat, lon))

        self.frame_count += 1

        # 🔥 FIX COLOR ISSUE
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return av.VideoFrame.from_ndarray(img, format="rgb24")


# 🔥 FORCE DEFAULT CAMERA (NO DROPDOWN)
webrtc_ctx = webrtc_streamer(
    key="pothole-live",
    video_processor_factory=LiveProcessor,

    media_stream_constraints={
        "video": {
            "facingMode": "user",   # 👈 FORCE LAPTOP CAMERA
            "width": {"ideal": 640},
            "height": {"ideal": 480}
        },
        "audio": False,
    },

    async_processing=True
)

# =====================================================
# 📊 SECTION 3 — LIVE ANALYTICS DASHBOARD
# =====================================================
if webrtc_ctx.video_processor:

    vp = webrtc_ctx.video_processor

    if len(vp.unique_detections) > 0:

        st.header("📊 Live Route Analysis")

        df = pd.DataFrame(vp.unique_detections)

        summary, total, risk = generate_summary(vp.unique_detections)

        st.write(summary)
        st.success(f"Total Unique Defects: {total}")

        show_risk_indicator(risk)

        # Dashboard
        st.subheader("📈 Live Dashboard")
        live_dashboard(df)

        # Time Series
        st.subheader("📉 Time Series")
        plot_time_series(df)

        # Distribution
        st.subheader("📊 Distribution")
        plot_bar_chart(summary)
        plot_pie_chart(summary)

        # Save
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/live_report.csv", index=False)

        st.download_button(
            "📥 Download CSV Report",
            df.to_csv(index=False),
            "road_report.csv"
        )

        # Map
        st.subheader("🗺️ Pothole Map")
        m = create_map(vp.map_points)

        if m:
            st.components.v1.html(m._repr_html_(), height=400)

        # Heatmap
        st.subheader("🔥 Heatmap")
        plot_heatmap(vp.map_points)

# =====================================================
# 📱 MOBILE SUPPORT
# =====================================================
st.header("📱 Mobile Support")

st.info("""
✔ Works on mobile browsers  
✔ Use Chrome for best performance  
✔ Allow camera permissions  
✔ HTTPS required for deployment  
""")