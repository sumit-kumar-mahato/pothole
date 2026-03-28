import sys
import os
import streamlit as st

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide")

import gdown
import cv2
import tempfile
import numpy as np
import folium
import pandas as pd
import time

# =====================================================
# 🔧 PATH + MODEL SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "best.pt")
FILE_ID = "1J391ph0-jzdI8vMh5WwZKgqZCl6wUenh"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =====================================================
# 📥 MODEL LOADER (CLEAN)
# =====================================================
@st.cache_resource
def load_detector():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    from utils.detector import PotholeDetector
    return PotholeDetector(MODEL_PATH)

# ✅ Load model safely
if "detector" not in st.session_state:
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading AI Model... ⏳")

    with st.spinner("Loading AI Model..."):
        st.session_state.detector = load_detector()

    st.success("Model Ready ✅")

detector = st.session_state.detector

# =====================================================
# 📦 IMPORT UTILS
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
# 🎨 UI
# =====================================================
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
# 📂 IMAGE / VIDEO UPLOAD
# =====================================================
st.header("📂 Upload Image / Video")

uploaded_file = st.file_uploader(
    "Upload road image or video",
    type=["jpg", "jpeg", "png", "mp4"]
)

# ---------------- IMAGE ----------------
def process_image(file):
    file_bytes = file.read()
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
def process_video(file):

    st.subheader("🎥 Video Processing")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    tracker = SimpleTracker()
    unique_detections = []

    if st.button("▶️ Start Video Analysis"):

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

# Run upload
if uploaded_file:
    if uploaded_file.type.startswith("image"):
        process_image(uploaded_file)
    elif uploaded_file.type.startswith("video"):
        process_video(uploaded_file)

# =====================================================
# 🎥 LIVE CAMERA
# =====================================================
st.header("🎥 Live Detection (Server Camera)")

start_camera = st.checkbox("Start Live Camera")

FRAME_WINDOW = st.image([])
tracker = SimpleTracker()
unique_detections = []
map_points = []

if start_camera:

    cap = cv2.VideoCapture(0)
    frame_count = 0

    while start_camera:
        ret, frame = cap.read()

        if not ret:
            st.error("Camera not working")
            break

        frame = cv2.resize(frame, (640, 480))

        frame, detections, stats, fps = detector.detect(frame)

        lat = BASE_LAT + frame_count * 0.00005
        lon = BASE_LON + frame_count * 0.00005

        for d in detections:
            if tracker.is_new(d["bbox"]):
                unique_detections.append(d)

                if d["label"] == "pothole":
                    map_points.append((lat, lon))

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_count += 1
        time.sleep(0.03)

    cap.release()

# =====================================================
# 📊 LIVE ANALYTICS
# =====================================================
if len(unique_detections) > 0:

    st.header("📊 Live Route Analysis")

    df = pd.DataFrame(unique_detections)

    summary, total, risk = generate_summary(unique_detections)

    st.write(summary)
    st.success(f"Total Unique Defects: {total}")

    show_risk_indicator(risk)

    st.subheader("📈 Dashboard")
    live_dashboard(df)

    st.subheader("📉 Time Series")
    plot_time_series(df)

    st.subheader("📊 Distribution")
    plot_bar_chart(summary)
    plot_pie_chart(summary)

    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/live_report.csv", index=False)

    st.download_button(
        "📥 Download CSV Report",
        df.to_csv(index=False),
        "road_report.csv"
    )

    st.subheader("🗺️ Map")
    m = create_map(map_points)

    if m:
        st.components.v1.html(m._repr_html_(), height=400)

    st.subheader("🔥 Heatmap")
    plot_heatmap(map_points)

# =====================================================
# 📱 MOBILE INFO
# =====================================================
st.header("📱 Mobile Support")

st.info("""
✔ Image & video detection works on mobile  
❌ Live camera requires local system (not browser)  
✔ Use Chrome for best performance  
""")
