import os, json, io, base64
from collections import Counter
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2

# 从 rubiks_core 加载求解逻辑
try:
    from rubiks_core import solve_cube
except ImportError:
    st.error("❌ 找不到 rubiks_core.py，请确保该文件在项目目录下。")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Rubik's AI Solver", page_icon="🧊",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
# CSS (UI 优化与防跑版)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    font-family:'Outfit',sans-serif!important;
    background: radial-gradient(circle at 0% 0%, #f8fafc 0%, #e2e8f0 100%)!important;
}
[data-testid="stHeader"] { background-color: transparent !important; }
[data-testid="stMainBlockContainer"]{ padding-top: 50px !important; }
.app-title { font-size: 2.8rem; font-weight: 800; color: #0f172a; margin-bottom: 0.2rem; }
.app-subtitle { font-size: 1.1rem; color: #64748b; margin-bottom: 2rem; }
.mcard{
    background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.4); border-radius: 28px; padding: 32px;
}
.stButton>button{ border-radius: 12px!important; font-weight: 800!important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
FACES = ['Up','Left','Front','Right','Back','Down']
HEX_COLORS = {'White':'#f1f5f9','Red':'#ef4444','Green':'#22c55e', 'Yellow':'#eab308','Orange':'#f97316','Blue':'#3b82f6'}
COLOR_EMOJIS = {'White':'⬜','Red':'🟥','Green':'🟩','Yellow':'🟨','Orange':'🟧','Blue':'🟦'}
CENTER_COLORS = {'Up':'White','Left':'Orange','Front':'Green', 'Right':'Red','Back':'Blue','Down':'Yellow'}
CALIB_FILE = "calibration_profile.json"

_DEFAULTS = {
    'active_face': 'Front',
    'cube_state': {f: (['White']*4+[CENTER_COLORS[f]]+['White']*4) for f in FACES},
    'last_solution': None,
    'custom_std_colors': {},
    'history': None,
    'history_index': 0,
    'confirmed_faces': [],
    'scan_result': None,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

if st.session_state.history is None:
    st.session_state.history = [json.dumps({"cube_state": st.session_state.cube_state, "confirmed_faces": st.session_state.confirmed_faces})]

def push_history():
    sj = json.dumps({"cube_state": st.session_state.cube_state, "confirmed_faces": st.session_state.confirmed_faces})
    st.session_state.history.append(sj)
    st.session_state.history_index = len(st.session_state.history) - 1

# ══════════════════════════════════════════════════════════════════════════════
# OPENCV & AI ROUTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_std_colors():
    d = {'White':(0,30,220),'Yellow':(30,160,200),'Orange':(12,200,240), 'Red':(0,210,180),'Green':(60,180,150),'Blue':(110,180,160)}
    for k, v in st.session_state.custom_std_colors.items(): d[k] = tuple(v)
    return d

def classify_color_lab(bgr_pixel, std_colors):
    pixel_mat = np.uint8([[bgr_pixel]])
    lab_pixel = cv2.cvtColor(pixel_mat, cv2.COLOR_BGR2LAB)[0][0]
    min_dist = float('inf'); best_color = 'White'
    for color_name, hsv_val in std_colors.items():
        std_bgr = cv2.cvtColor(np.uint8([[[hsv_val[0], hsv_val[1], hsv_val[2]]]]), cv2.COLOR_HSV2BGR)
        std_lab = cv2.cvtColor(std_bgr, cv2.COLOR_BGR2LAB)[0][0]
        dist = np.linalg.norm(lab_pixel.astype(float) - std_lab.astype(float))
        if dist < min_dist:
            min_dist = dist; best_color = color_name
    return best_color

def _grid_colors_with_pixels(warped, std_colors, classifier_fn):
    detected = ['White']*9; raw_bgrs = [np.zeros(3, dtype=np.uint8)]*9
    for r in range(3):
        for c in range(3):
            tx, ty = int((c+.5)*100), int((r+.5)*100)
            roi = warped[max(0,ty-8):min(300,ty+8), max(0,tx-8):min(300,tx+8)]
            if roi.size > 0: bgr = np.median(roi, axis=(0,1)).astype(np.uint8)
            else: bgr = np.zeros(3, dtype=np.uint8)
            detected[r*3+c] = classifier_fn(bgr)
            raw_bgrs[r*3+c] = bgr
    return detected, raw_bgrs

def run_method_a(raw_bytes):
    # Method A: Standard OpenCV
    arr = np.frombuffer(raw_bytes, dtype=np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    warped = cv2.resize(img, (300, 300)) # Simplified for example
    std = get_std_colors()
    det, raw_bgrs = _grid_colors_with_pixels(warped, std, lambda b: classify_color_lab(b, std))
    return det, raw_bgrs, cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), None

def run_method_b(raw_bytes):
    # Method B: YOLOv8
    try:
        import yolo_detect
        result = yolo_detect.get_cube_bbox(raw_bytes, draw=True)
        if result:
            cropped_300 = cv2.resize(result["cropped"], (300, 300))
            std = get_std_colors()
            det, raw_bgrs = _grid_colors_with_pixels(cropped_300, std, lambda b: classify_color_lab(b, std))
            return det, raw_bgrs, cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB), None
        return None, None, None, "❌ YOLO failed to detect a cube."
    except ImportError:
        return None, None, None, "❌ Missing yolo_detect.py"

# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
def render_live_cube_map(active_face):
    cube = st.session_state.cube_state
    def face_html(f):
        c_html = "".join([f'<div style="width:20px;height:20px;background:{HEX_COLORS[cube[f][i]]};border-radius:2px;"></div>' for i in range(9)])
        border = "2px solid #6366f1" if f == active_face else "1px solid #ddd"
        return f'<div style="display:flex;flex-direction:column;align-items:center;"><div style="font-size:9px;">{f}</div><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:2px;border:{border};padding:2px;border-radius:4px;">{c_html}</div></div>'
    
    html = f'''<div style="display:grid;grid-template-columns:repeat(4,70px);gap:10px;justify-content:center;font-family:sans-serif;">
        <div style="grid-column:2;">{face_html('Up')}</div>
        <div style="grid-row:2;">{face_html('Left')}</div><div style="grid-row:2;">{face_html('Front')}</div><div style="grid-row:2;">{face_html('Right')}</div><div style="grid-row:2;">{face_html('Back')}</div>
        <div style="grid-column:2;grid-row:3;">{face_html('Down')}</div>
    </div>'''
    components.html(html, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP LOGIC
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="app-title">🧊 AI Rubik\'s Vision Engine</div>', unsafe_allow_html=True)

if app_mode == "🧩 Scan & Solve":
    curr = st.session_state.active_face
    col_l, col_r, col_map = st.columns([3, 2, 2])

    with col_l:
        up = st.file_uploader(f"Upload {curr} Face", type=['jpg','png','jpeg'])
        algo = st.selectbox("Vision Engine", ["OpenCV", "YOLOv8"])
        
        if up:
            raw = up.getvalue()
            scan_key = f"scanned_{curr}_{algo}"
            if scan_key not in st.session_state:
                with st.spinner("Analyzing..."):
                    det, raw_bgrs, overlay, err = run_method_a(raw) if algo == "OpenCV" else run_method_b(raw)
                    if not err:
                        det[4] = CENTER_COLORS[curr]
                        st.session_state.cube_state[curr] = det
                        st.session_state.scan_result = {"overlay": overlay, "face": curr}
                        st.session_state[scan_key] = True
                        push_history(); st.rerun()
            
            if st.session_state.scan_result:
                st.image(st.session_state.scan_result["overlay"], use_container_width=True)

    with col_r:
        st.write("✏️ Manual Edit")
        # Grid buttons logic here (omitted for brevity, keep your existing grid code)

    with col_map:
        render_live_cube_map(curr)

    # Solve Button
    if st.button("⚡ Solve Cube", type="primary"):
        success, res = solve_cube(st.session_state.cube_state)
        if success: st.success(f"Solution: {res}")
        else: st.error(res)
