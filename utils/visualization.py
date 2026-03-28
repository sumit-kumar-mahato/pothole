import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

# ================= BAR =================
def plot_bar_chart(summary):
    df = pd.DataFrame(list(summary.items()), columns=["Type", "Count"])
    st.bar_chart(df.set_index("Type"))

# ================= PIE =================
def plot_pie_chart(summary):
    labels = list(summary.keys())
    sizes = list(summary.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    st.pyplot(fig)

# ================= RISK =================
def show_risk_indicator(risk):
    if risk == "Low":
        st.success(f"🟢 Risk Level: {risk}")
    elif risk == "Medium":
        st.warning(f"🟡 Risk Level: {risk}")
    else:
        st.error(f"🔴 Risk Level: {risk}")


# =====================================================
# 🔥 LIVE DASHBOARD
# =====================================================
def live_dashboard(df):

    st.subheader("📊 Live Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Defects", len(df))
    col2.metric("Potholes", len(df[df["label"] == "pothole"]))
    col3.metric("Cracks", len(df[df["label"] != "pothole"]))


# =====================================================
# 🔥 TIME SERIES
# =====================================================
def plot_time_series(df):

    if "frame" not in df.columns:
        return

    st.subheader("⏱ Time-Series Analysis")

    df_grouped = df.groupby("frame").size()

    st.line_chart(df_grouped)


# =====================================================
# 🔥 HEATMAP
# =====================================================
def plot_heatmap(map_points):

    if not map_points:
        return None

    st.subheader("🔥 Pothole Heatmap")

    m = folium.Map(location=map_points[0], zoom_start=14)

    HeatMap(map_points).add_to(m)

    st.components.v1.html(m._repr_html_(), height=400)