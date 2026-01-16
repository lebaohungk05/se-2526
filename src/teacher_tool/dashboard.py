import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

# Set Page Config
st.set_page_config(page_title="Engagement Dashboard", layout="wide")

st.title("🎓 Student Engagement Analysis Dashboard")
st.markdown("Hệ thống báo cáo mức độ tập trung của sinh viên sau buổi học.")

# 1. Load Data
REPORT_DIR = "reports"
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

files = glob.glob(os.path.join(REPORT_DIR, "*.csv"))
files.sort(key=os.path.getmtime, reverse=True)

if not files:
    st.warning("Chưa có dữ liệu báo cáo. Hãy chạy Teacher Tool trước!")
    st.stop()

# Sidebar: Select Session
selected_file = st.sidebar.selectbox("Chọn Buổi Học (File Log)", files)
st.sidebar.markdown("---")
st.sidebar.info(f"Đang xem: {os.path.basename(selected_file)}")

# Read CSV
df = pd.read_csv(selected_file)

# 2. Key Metrics
col1, col2, col3 = st.columns(3)
avg_focus = df['avg_engagement'].mean()
max_students = df['face_count'].max()
distracted_frames = df[df['status'] == 'DISTRACTED'].shape[0]
total_frames = df.shape[0]
distracted_pct = (distracted_frames / total_frames * 100) if total_frames > 0 else 0

col1.metric("Chỉ số Tập trung TB", f"{avg_focus:.2f}", delta_color="normal")
col2.metric("Số SV tham gia (Max)", int(max_students))
col3.metric("Tỉ lệ Mất tập trung", f"{distracted_pct:.1f}%", delta_color="inverse")

# 3. Charts
st.markdown("### 1. Biến động sự tập trung theo thời gian")
fig_line = px.line(df, x='timestamp', y='avg_engagement', 
                   title='Average Class Engagement over Time',
                   markers=True)
# Add threshold lines
fig_line.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="High Focus")
fig_line.add_hline(y=-0.2, line_dash="dash", line_color="red", annotation_text="Distracted")
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("### 2. Phân bố Cảm xúc (Tổng hợp)")
# Sum up all emotion columns
emotion_cols = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# Check if columns exist (for old CSV compatibility)
valid_cols = [c for c in emotion_cols if c in df.columns]

if valid_cols:
    total_emotions = df[valid_cols].sum().reset_index()
    total_emotions.columns = ['Emotion', 'Count']
    
    fig_pie = px.pie(total_emotions, values='Count', names='Emotion', 
                     title='Overall Emotion Distribution',
                     color='Emotion',
                     color_discrete_map={
                         'Happy': '#2ecc71',
                         'Surprise': '#f1c40f',
                         'Neutral': '#95a5a6',
                         'Sad': '#3498db',
                         'Angry': '#e74c3c',
                         'Fear': '#8e44ad',
                         'Disgust': '#d35400'
                     })
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### 3. Chi tiết Cảm xúc theo Thời gian (Area Chart)")
    fig_area = px.area(df, x='timestamp', y=valid_cols, 
                       title='Detailed Emotion Trends',
                       color_discrete_map={
                         'Happy': '#2ecc71',
                         'Surprise': '#f1c40f',
                         'Neutral': '#95a5a6',
                         'Sad': '#3498db',
                         'Angry': '#e74c3c',
                         'Fear': '#8e44ad',
                         'Disgust': '#d35400'
                       })
    st.plotly_chart(fig_area, use_container_width=True)
else:
    st.warning("File log cũ không chứa thông tin chi tiết cảm xúc. Hãy chạy lại Tool mới.")

# 4. Data Table
with st.expander("Xem dữ liệu chi tiết (Raw Data)"):
    st.dataframe(df)
