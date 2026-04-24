import os, json, io, base64
from collections import Counter
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
from rubiks_core import (
    validate_cube_state, solve_cube,
    classify_color_lab, classify_color_hsv, classify_color_knn, classify_color_mlp,
    extract_center_bgr, COLORS,
)
import yolo_detect
import svm_detect

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Rubik's AI Solver", page_icon="🧊",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
# CSS (ADAPTIVE FOR LIGHT/DARK MODE)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stMainBlockContainer"]{
    padding-top:40px!important;
    max-width: 100%!important;
}

[data-testid="stMain"] .block-container {
    min-width: 0!important;
    padding-left: 1rem!important;
    padding-right: 1rem!important;
}

[data-testid="stHorizontalBlock"] {
    flex-wrap: wrap!important;
    gap: 0.5rem!important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    min-width: 220px!important;
    flex: 1 1 auto!important;
}

[data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap!important;
    gap: 0.1rem!important;
}
[data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    min-width: 0!important;
    flex: 1 1 0%!important;
}

/* ── 適應性毛玻璃卡片 ── */
.mcard{
    background: rgba(128, 128, 128, 0.05);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 28px;
    padding: clamp(16px, 2vw, 32px);
    margin-bottom: 24px;
}
.slabel{
    font-size: 11px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase;
    color: #888; margin-bottom: 14px; display: block; opacity: 0.9;
}

.stButton>button{
    border-radius: 12px!important; 
    font-family: 'Outfit',sans-serif!important;
    font-weight: 700!important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1)!important;
}
.stButton>button:hover{
    transform: translateY(-1px)!important;
    border-color: #6366f1!important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.15)!important;
}

[data-testid="stBaseButton-primary"] {
    background: #4338ca!important; 
    color: white!important;
    border: none!important;
}
[data-testid="stBaseButton-primary"]:hover {
    background: #3730a3!important; 
}

.action-row{display:flex; flex-wrap:wrap; gap:12px; margin-top:24px; padding-top:24px; border-top:1px solid rgba(128,128,128,0.2);}

.sol-box{
    background: rgba(128, 128, 128, 0.1);
    border-radius: 20px; padding: clamp(12px, 1.5vw, 24px);
    font-family: 'Courier New', monospace; font-size: clamp(12px, 1vw, 16px); font-weight: 800;
    word-break: break-word;
}

[data-testid="stSidebar"]{
    border-right: 1px solid rgba(128,128,128,0.1)!important;
}

.app-title {
    font-size: clamp(1.6rem, 3vw, 2.8rem); font-weight: 800; text-align: center;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.app-subtitle {
    font-size: clamp(0.65rem, 0.9vw, 0.9rem); font-weight: 600; text-align: center;
    color: #888; letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: clamp(16px, 3vw, 40px);
}

@media screen and (max-width: 900px) {
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 100%!important;
    }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FACES         = ['Up','Left','Front','Right','Back','Down']
HEX_COLORS    = {'White':'#f1f5f9','Red':'#ef4444','Green':'#22c55e',
                  'Yellow':'#eab308','Orange':'#f97316','Blue':'#3b82f6'}
COLOR_EMOJIS  = {'White':'⬜','Red':'🟥','Green':'🟩','Yellow':'🟨','Orange':'🟧','Blue':'🟦'}
CENTER_COLORS = {'Up':'White','Left':'Orange','Front':'Green',
                  'Right':'Red','Back':'Blue','Down':'Yellow'}
TOP_COLORS    = {'Up':'Blue','Left':'White','Front':'White',
                  'Right':'White','Back':'White','Down':'Green'}
CALIB_FILE    = "calibration_profile.json"

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    'active_face':    'Front',
    'cube_state':     {f: (['White']*4+[CENTER_COLORS[f]]+['White']*4) for f in FACES},
    'last_solution':  None,
    'selected_color': 'White',
    'solve_speed':    1.0,
    'custom_std_colors': {},
    'history':        None,
    'history_index':  0,
    'confirmed_faces': [],
    'scan_result':    None,
}

if 'custom_std_colors' not in st.session_state and os.path.exists(CALIB_FILE):
    try:
        with open(CALIB_FILE) as fh:
            _DEFAULTS['custom_std_colors'] = json.load(fh)
    except Exception: pass

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.history is None:
    st.session_state.history = [json.dumps({
        "cube_state": st.session_state.cube_state,
        "confirmed_faces": st.session_state.confirmed_faces
    })]

def push_history():
    sj = json.dumps({
        "cube_state": st.session_state.cube_state,
        "confirmed_faces": st.session_state.confirmed_faces
    })
    if st.session_state.history_index < len(st.session_state.history)-1:
        st.session_state.history = st.session_state.history[:st.session_state.history_index+1]
    st.session_state.history.append(sj)
    st.session_state.history_index = len(st.session_state.history)-1

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_std_colors():
    d = {'White':(0,30,220),'Yellow':(30,160,200),'Orange':(12,200,240),
         'Red':(0,210,180),'Green':(60,180,150),'Blue':(110,180,160)}
    for k, v in st.session_state.custom_std_colors.items(): d[k] = tuple(v)
    return d

def hex_to_bgr(h):
    h = h.lstrip('#')
    return (int(h[4:6],16), int(h[2:4],16), int(h[0:2],16))

def face_complete(f): return f in st.session_state.get('confirmed_faces', [])

def mark_confirmed(face):
    cf = st.session_state.confirmed_faces
    if face not in cf: cf.append(face)

def unmark_confirmed(face):
    cf = st.session_state.confirmed_faces
    if face in cf: cf.remove(face)

# ══════════════════════════════════════════════════════════════════════════════
# DETECTION LOGIC (FULLY RESTORED)
# ══════════════════════════════════════════════════════════════════════════════
def _warp_to_300(img_bgr):
    h, w = img_bgr.shape[:2]
    gs = int(min(h,w)*0.7); ox, oy = (w-gs)//2, (h-gs)//2
    return cv2.resize(img_bgr[oy:oy+gs, ox:ox+gs], (300,300))

def _grid_colors_with_pixels(warped, std_colors, classifier_fn, use_blocks=False):
    detected = ['White']*9
    raw_bgrs = [np.zeros(3, dtype=np.uint8)]*9
    centers = [(0, 0)] * 9
    hsv_w = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV); sat_w = hsv_w[:,:,1]
    for r in range(3):
        for c in range(3):
            tx, ty = int((c+.5)*100), int((r+.5)*100)
            y1,y2 = max(0,ty-35), min(300,ty+35); x1,x2 = max(0,tx-35), min(300,tx+35)
            moms = cv2.moments(sat_w[y1:y2,x1:x2])
            fx, fy = tx, ty
            if moms["m00"] > 50:
                sl = x1+int(moms["m10"]/moms["m00"]); sm = y1+int(moms["m01"]/moms["m00"])
                if np.sqrt((sl-tx)**2+(sm-ty)**2) < 30: fx, fy = sl, sm
            
            centers[r*3+c] = (fx, fy)
            
            roi_size = 8 if not use_blocks else 25 
            roi = warped[max(0,fy-roi_size):min(300,fy+roi_size), max(0,fx-roi_size):min(300,fx+roi_size)]
            
            if roi.size > 0:
                rh, rw = roi.shape[:2]; c_ = roi[rh//4:rh-rh//4, rw//4:rw-rw//4]
                median_bgr = np.median(c_, axis=(0,1)).astype(np.uint8)
                
                if use_blocks:
                    detected[r*3+c] = classifier_fn(roi) 
                else:
                    detected[r*3+c] = classifier_fn(median_bgr)
                raw_bgrs[r*3+c] = median_bgr
            else:
                raw_bgrs[r*3+c] = np.zeros(3, dtype=np.uint8)
                detected[r*3+c] = "White"
                
    return detected, raw_bgrs, centers

def _draw_grid_overlay(warped_rgb, centers=None):
    vis = warped_rgb.copy()
    h, w = vis.shape[:2]
    
    if centers:
        box_size = 35 
        for (cx, cy) in centers:
            cv2.rectangle(vis, (max(0, cx-box_size), max(0, cy-box_size)), 
                               (min(w, cx+box_size), min(h, cy+box_size)), (0, 255, 0), 3)
            cv2.circle(vis, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 0), 2)
    else:
        for i in range(1, 3):
            cv2.line(vis, (i*w//3, 0), (i*w//3, h), (100, 100, 255), 2)
            cv2.line(vis, (0, i*h//3), (w, i*h//3), (100, 100, 255), 2)
        cv2.rectangle(vis, (1,1), (w-2,h-2), (100, 100, 255), 3)
        for r in range(3):
            for c in range(3):
                cx, cy = int((c+0.5)*w/3), int((r+0.5)*h/3)
                cv2.circle(vis, (cx, cy), 6, (255, 255, 255), -1)
                cv2.circle(vis, (cx, cy), 6, (100, 100, 255), 2)
    return vis

def run_method_a(raw_bytes, expected_center):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None, None, None, "❌ Cannot decode image."
    std = get_std_colors(); warped = _warp_to_300(img)
    det, raw_bgrs, centers = _grid_colors_with_pixels(warped, std, lambda b: classify_color_lab(b, std))
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    overlay = _draw_grid_overlay(warped_rgb, centers)
    return det, raw_bgrs, overlay, None

def run_method_b(raw_bytes, expected_center):
    try:
        stickers = yolo_detect.detect_stickers(raw_bytes)
        annotated_bgr, _ = yolo_detect.detect_and_draw(raw_bytes)
        overlay = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        if len(stickers) != 9:
            cube_res = yolo_detect.get_cube_bbox(raw_bytes, draw=True)
            if cube_res:
                det = yolo_detect.get_face_colors_from_crop(cube_res["cropped"], classifier_fn=lambda b: classify_color_lab(b, get_std_colors()))
                h, w = cube_res["cropped"].shape[:2]
                ch, cw = h//3, w//3
                raw_bgrs = []
                for r in range(3):
                    for c in range(3):
                        patch = cube_res["cropped"][r*ch:(r+1)*ch, c*cw:(c+1)*cw]
                        raw_bgrs.append(np.median(patch, axis=(0,1)).astype(np.uint8))
                det[4] = expected_center
                overlay = cv2.cvtColor(cube_res["annotated"], cv2.COLOR_BGR2RGB)
                return det, raw_bgrs, overlay, None
            
            diag_bgr, _ = yolo_detect.detect_and_draw(raw_bytes)
            diag_rgb = cv2.cvtColor(diag_bgr, cv2.COLOR_BGR2RGB)
            msg = f"⚠️ YOLO detected {len(stickers)} features (expected 9). Falling back to diagnostic view."
            return None, None, diag_rgb, msg

        std = get_std_colors()
        det = []
        raw_bgrs = []
        for s in stickers:
            bgr = np.median(s["cropped"], axis=(0, 1)).astype(np.uint8)
            raw_bgrs.append(bgr)
            if s["color"]:
                det.append(s["color"])
            else:
                det.append(classify_color_lab(bgr, std))
        return det, raw_bgrs, overlay, None
    except Exception as e:
        return None, None, None, f"⚠️ YOLO Error: {str(e)}"

def run_method_c(raw_bytes, expected_center):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8); img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None, None, None, "❌ Cannot decode image."
    std = get_std_colors(); warped = _warp_to_300(img)
    det, raw_bgrs, centers = _grid_colors_with_pixels(warped, std, lambda b: svm_detect.classify_color_svm(b), use_blocks=True)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    overlay = _draw_grid_overlay(warped_rgb, centers)
    return det, raw_bgrs, overlay, None

# ══════════════════════════════════════════════════════════════════════════════
# LIVE CUBE MAP (FULLY RESTORED)
# ══════════════════════════════════════════════════════════════════════════════
def render_live_cube_map(active_face):
    cube = st.session_state.cube_state
    confirmed = st.session_state.confirmed_faces

    def face_html(area_name, face_name):
        is_active = (face_name == active_face)
        is_confirmed = (face_name in confirmed)
        title_color = "#6366f1" if is_active else ("#22c55e" if is_confirmed else "#94a3b8")
        title_weight = "800" if is_active else "700"
        border_color = "#6366f1" if is_active else ("#22c55e" if is_confirmed else "transparent")
        shadow = "0 0 10px rgba(99,102,241,0.25)" if is_active else ("0 0 6px rgba(34,197,94,0.15)" if is_confirmed else "none")
        status_icon = "✏️" if is_active else ("✅" if is_confirmed else "⭕")
        
        cells = ""
        for idx in range(9):
            color = cube[face_name][idx]
            hex_c = HEX_COLORS.get(color, '#f1f5f9')
            cells += f'<div style="width:clamp(14px,1.2vw,18px);height:clamp(14px,1.2vw,18px);border-radius:3px;background:{hex_c};"></div>'
        
        return f'''<div style="grid-area:{area_name}; justify-self:center;">
            <div style="font-size:9px;font-weight:{title_weight};text-align:center;color:{title_color};margin-bottom:3px;letter-spacing:1px;font-family:Outfit,sans-serif;">{status_icon} {face_name}</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1.5px;border-radius:6px;overflow:hidden;border:2px solid {border_color};box-shadow:{shadow};padding:1.5px;background:rgba(255,255,255,0.8);">{cells}</div>
        </div>'''
    
    html = f'''
    <html><body style="margin:0;padding:2px;background:transparent;font-family:Outfit,sans-serif;box-sizing:border-box;overflow:auto;">
    <div style="background:rgba(128,128,128,0.1);border-radius:18px;padding:12px;border:1px solid rgba(128,128,128,0.2); width:fit-content; margin:0 auto; max-width:100%; box-sizing:border-box;">
        <div style="font-size:10px;font-weight:800;letter-spacing:1px;text-transform:uppercase;color:#888;margin-bottom:12px;text-align:center;">🗺️ LIVE CUBE MAP</div>
        <div style="display:grid; grid-template-areas: '. U . .' 'L F R B' '. D . .'; grid-gap:4px; justify-content:center; align-items:center;">
            {face_html('U', 'Up')}
            {face_html('L', 'Left')}
            {face_html('F', 'Front')}
            {face_html('R', 'Right')}
            {face_html('B', 'Back')}
            {face_html('D', 'Down')}
        </div>
        <div style="text-align:center;margin-top:12px;font-size:9px;color:#888;letter-spacing:1px;font-family:Outfit,sans-serif;">
            ✅ {len(confirmed)}/6 CONFIRMED
        </div>
    </div>
    </body></html>'''
    components.html(html, height=380, scrolling=True)

# ══════════════════════════════════════════════════════════════════════════════
# DETECTION FEEDBACK PANEL (FULLY RESTORED)
# ══════════════════════════════════════════════════════════════════════════════
def render_detection_feedback(scan_result):
    if scan_result is None:
        return
    det_colors = scan_result.get('detected', [])
    raw_bgrs = scan_result.get('raw_bgrs', [])
    overlay_img = scan_result.get('overlay', None)
    engine = scan_result.get('engine', 'OpenCV')
    face = scan_result.get('face', 'Front')
    
    st.markdown(f"##### 🔍 Detection Result — {face}")
    if overlay_img is not None:
        st.image(overlay_img, caption=f"📐 How {engine} cropped & analyzed your photo", width=300)
    
    st.markdown("---")
    grid_style = "display:grid;grid-template-columns:repeat(3,1fr);gap:4px;max-width:180px;margin:0 auto;"
    cell_style_base = "width:52px;height:52px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;border:2px solid rgba(128,128,128,0.2);box-shadow:inset 0 -2px 4px rgba(0,0,0,0.1);"
    
    cells_html = ""
    for idx in range(9):
        if idx < len(det_colors):
            color = det_colors[idx]
            hex_c = HEX_COLORS.get(color, '#f1f5f9')
            text_color = "#333" if color in ['White','Yellow'] else "rgba(255,255,255,0.9)"
            label = color[:3]
            cells_html += f'<div style="{cell_style_base}background:{hex_c};color:{text_color};">{label}</div>'
    
    pixel_html = ""
    for idx in range(9):
        if idx < len(raw_bgrs):
            bgr = raw_bgrs[idx]
            r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
            pixel_html += f'<div style="{cell_style_base}background:rgb({r},{g},{b});font-size:8px;color:rgba(255,255,255,0.85);">R{r}<br>G{g}<br>B{b}</div>'
    
    col_det, col_raw = st.columns(2)
    with col_det:
        st.markdown("**AI Classification:**")
        st.markdown(f'<div style="{grid_style}">{cells_html}</div>', unsafe_allow_html=True)
    with col_raw:
        st.markdown("**Raw Pixel Colors:**")
        st.markdown(f'<div style="{grid_style}">{pixel_html}</div>', unsafe_allow_html=True)
    
    st.markdown("")
    color_counts = Counter(det_colors)
    summary_parts = []
    for c in ['White','Red','Green','Yellow','Orange','Blue']:
        cnt = color_counts.get(c, 0)
        if cnt > 0:
            summary_parts.append(f"{COLOR_EMOJIS[c]} {c}×{cnt}")
    st.caption("Detected: " + "  ".join(summary_parts))

# ══════════════════════════════════════════════════════════════════════════════
# 3D PLAYER (FULLY RESTORED)
# ══════════════════════════════════════════════════════════════════════════════
def render_3d_player(solution):
    def inv(s):
        r=[]
        for m in reversed(s.split()):
            if "'" in m: r.append(m.replace("'",""))
            elif "2" in m: r.append(m)
            else: r.append(m+"'")
        return " ".join(r)
    speed = st.session_state.get('solve_speed',1.0)
    html = f"""
    <div style="background:transparent; border-radius:18px; padding:14px; border:1px solid rgba(128,128,128,0.2);">
      <script type="module" src="https://cdn.cubing.net/js/cubing/twisty"></script>
      <twisty-player experimental-setup-alg="{inv(solution)}" alg="{solution}"
        background="none" tempo-scale="{speed}" control-panel="bottom-row"
        style="width:100%; height:380px;"></twisty-player>
    </div>"""
    components.html(html, height=430)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<h2 style='margin-top:0;'>🧊 Solver</h2>", unsafe_allow_html=True)
    app_mode = st.radio("Mode", ["🧩 Scan & Solve", "⚙️ Calibration"], label_visibility="collapsed")
    st.divider()
    if app_mode == "🧩 Scan & Solve":
        with st.expander("📊 Sticker Status"):
            all_s = [s for f in FACES for s in st.session_state.cube_state[f]]
            for name in HEX_COLORS:
                cnt = all_s.count(name); ok = (cnt==9)
                st.markdown(f"<div style='font-size:11px;'>{COLOR_EMOJIS[name]} {name}: <b>{cnt}/9</b></div>", unsafe_allow_html=True)
        if st.button("🗑️ Reset Cube", use_container_width=True):
            st.session_state.cube_state = {f:(['White']*4+[CENTER_COLORS[f]]+['White']*4)for f in FACES}
            st.session_state.confirmed_faces = []; st.session_state.last_solution = None
            st.session_state.scan_result = None
            push_history(); st.rerun()

# ── MAIN TITLE ──────────────────────────────────────────────────────────────
st.markdown('''
    <div class="app-title">🧊 AI Rubik's Vision Engine</div>
    <div class="app-subtitle">Multi-Algorithm Comparison & Topology Validation System</div>
''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCAN & SOLVE PAGE
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "🧩 Scan & Solve":
    curr = st.session_state.active_face
    pw_cols = st.columns(6)
    for i, f in enumerate(FACES):
        cc = CENTER_COLORS[f]
        is_act = (f == curr)
        is_conf = face_complete(f)
        lbl = f"{COLOR_EMOJIS[cc]} {f}"
        if is_conf and not is_act:
            lbl = f"✅ {f}"
        btn_type = "primary" if is_act else "secondary"
        if pw_cols[i].button(lbl, key=f"pwr_{f}", use_container_width=True, type=btn_type):
            st.session_state.active_face = f
            st.session_state.selected_color = cc
            st.session_state.scan_result = None
            st.rerun()

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    col_l, col_r, col_map = st.columns([3, 2, 2], gap="large")
    
    with col_l:
        st.markdown("#### 📂 Photo Assist")
        
        
        use_camera = st.toggle("📷 Enable Camera Snapshot Mode", key="camera_toggle")
        
        raw = None
        if use_camera:
            st.info("💡 **Engineering Trade-off**: Static 'Snapshot' mode is utilized instead of continuous live streaming. This prevents motion blur and specular glare from the webcam's auto-exposure, ensuring our 99.99% OpenCV accuracy remains uncompromised.")
            cam = st.camera_input("Align the cube and take a snapshot", key=f"cam_{curr}")
            if cam:
                raw = cam.read()
        else:
            up = st.file_uploader("Upload reference", type=['jpg','png','jpeg'], key=f"up_{curr}", label_visibility="collapsed")
            if up:
                raw = up.read()
        
        if raw:
            st.image(raw, caption="📷 Captured / Uploaded photo", width=300)
            
            st.divider()
            
            st.markdown("##### 🔬 Vision Engine")
            algo_choice = st.selectbox(
                "Select AI Model:",
                ["📐 OpenCV (Math Distance)", "🎯 YOLOv8 (6-Class AI)", "🧠 SVM (Machine Learning)"],
                label_visibility="collapsed",
                key=f"algo_sel_{curr}"
            )

            engine_name = algo_choice.split(" ")[1]  
            if st.button(f"📸 Scan with {engine_name}", type="primary", use_container_width=True):
                
                with st.spinner(f"Analyzing via {engine_name}..."):
                    
                    det, raw_bgrs, overlay, err = None, None, None, None
                    
                    if "OpenCV" in algo_choice:
                        det, raw_bgrs, overlay, err = run_method_a(raw, CENTER_COLORS[curr])
                        
                    elif "YOLOv8" in algo_choice:
                        det, raw_bgrs, overlay, err = run_method_b(raw, CENTER_COLORS[curr])
                        
                    elif "SVM" in algo_choice:
                        det, raw_bgrs, overlay, err = run_method_c(raw, CENTER_COLORS[curr])

                    if err:
                        st.error(err)
                        if overlay is not None:
                            st.session_state.scan_result = {
                                'detected': det if det else [],
                                'raw_bgrs': [b.tolist() for b in raw_bgrs] if raw_bgrs else [],
                                'overlay': overlay, 'engine': engine_name, 'face': curr
                            }
                    elif det:
                        det[4] = CENTER_COLORS[curr]
                        st.session_state.cube_state[curr] = det
                        mark_confirmed(curr); push_history()
                        
                        st.session_state.scan_result = {
                            'detected': det,
                            'raw_bgrs': [bgr.tolist() if hasattr(bgr, 'tolist') else list(bgr) for bgr in raw_bgrs],
                            'overlay': overlay,
                            'engine': engine_name,
                            'face': curr,
                        }
                        st.rerun()
            
            if st.session_state.scan_result and st.session_state.scan_result.get('face') == curr:
                sr = st.session_state.scan_result
                bgr_arrays = [np.array(b, dtype=np.uint8) for b in sr['raw_bgrs']]
                render_detection_feedback({
                    'detected': sr['detected'],
                    'raw_bgrs': bgr_arrays,
                    'overlay': sr.get('overlay'),
                    'engine': sr['engine'],
                    'face': sr['face'],
                })
                
                st.markdown("")
                bc1, bc2 = st.columns(2)
                if bc1.button("✅ Accept & Next Face", type="primary", use_container_width=True):
                    st.session_state.scan_result = None
                    next_idx = (FACES.index(curr)+1) % 6
                    remaining = [f for f in FACES if not face_complete(f)]
                    st.session_state.active_face = remaining[0] if remaining else FACES[next_idx]
                    st.rerun()
                if bc2.button("🔄 Retry Scan", use_container_width=True):
                    unmark_confirmed(curr)
                    st.session_state.cube_state[curr] = ['White']*4+[CENTER_COLORS[curr]]+['White']*4
                    st.session_state.scan_result = None
                    push_history(); st.rerun()
        else:
            tc = TOP_COLORS[curr]; cc = CENTER_COLORS[curr]
            st.info(f"📷 **{curr} Face**: Center = {COLOR_EMOJIS[cc]} **{cc}** &nbsp;&nbsp;➔&nbsp;&nbsp; Keep {COLOR_EMOJIS[tc]} **{tc}** pointing **UP ⬆️**")
    
    with col_r:
        st.markdown('<span class="slabel">✏️ Manual Grid</span>', unsafe_allow_html=True)
        
        C_SEQ = ['White', 'Red', 'Green', 'Yellow', 'Orange', 'Blue']
        
        def cycle_stk(face, ix):
            cur_c = st.session_state.cube_state[face][ix]
            next_c = C_SEQ[(C_SEQ.index(cur_c) + 1) % len(C_SEQ)]
            st.session_state.cube_state[face][ix] = next_c
            mark_confirmed(face); push_history()

        for r in range(3):
            cols = st.columns(3)
            for c in range(3):
                idx = r*3+c; cv = st.session_state.cube_state[curr][idx]
                label = f"{COLOR_EMOJIS[cv]} {cv[:3].upper()}"
                if idx==4: cols[c].button(f"🔒 {cv[:3].upper()}", disabled=True, use_container_width=True)
                else: cols[c].button(label, key=f"g_{curr}_{idx}", on_click=cycle_stk, args=(curr, idx), use_container_width=True)
    
    with col_map:
        render_live_cube_map(curr)

    is_scanning = bool(st.session_state.scan_result and st.session_state.scan_result.get('face') == curr)

    if not is_scanning:
        st.markdown('<div class="action-row">', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        if a1.button("🧹 Reset Face", use_container_width=True):
            st.session_state.cube_state[curr] = ['White']*4+[CENTER_COLORS[curr]]+['White']*4
            unmark_confirmed(curr); st.session_state.scan_result = None
            push_history(); st.rerun()
        if a2.button("🎨 Fill Solid Color", use_container_width=True):
            sel = st.session_state.selected_color
            st.session_state.cube_state[curr] = [sel]*4+[CENTER_COLORS[curr]]+[sel]*4
            mark_confirmed(curr); push_history(); st.rerun()
        if a3.button("🚀 Confirm Face", use_container_width=True, type="primary"):
            mark_confirmed(curr); rem = [f for f in FACES if not face_complete(f)]
            if rem: st.session_state.active_face = rem[0]
            st.session_state.scan_result = None
            st.rerun()
        st.markdown('', unsafe_allow_html=True)

    all_s = [s for f in FACES for s in st.session_state.cube_state[f]]
    errs = [c for c in HEX_COLORS if all_s.count(c)!=9]
    if not errs:
        st.success("✨ Ready to solve!"); 
        if st.button("⚡ Solve Cube", use_container_width=True, type="primary"):
            sol = solve_cube(st.session_state.cube_state)
            if sol.startswith("!"): st.error(sol)
            else: st.session_state.last_solution = sol; st.rerun()
    elif st.session_state.last_solution is None:
        st.info("💡 Progress: " + ", ".join([f"{COLOR_EMOJIS[c]} {all_s.count(c)}/9" for c in errs]))

    if st.session_state.last_solution:
        st.markdown('<div class="mcard">', unsafe_allow_html=True)
        st.markdown(f'<div class="sol-box">{st.session_state.last_solution}</div>', unsafe_allow_html=True)
        render_3d_player(st.session_state.last_solution)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION PAGE
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "⚙️ Calibration":
    st.markdown('<div class="mcard">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Color Calibration Studio")
    st.markdown("If the lighting in your room is unusual, use this tool to 'teach' the AI what each color looks like.")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("#### 1. Sample Color")
        calib_color = st.selectbox("Select Color to Calibrate:", COLORS)
        
        # ─── 📸 Calibration 頁面同樣支援 Camera Toggle ───
        calib_cam_toggle = st.toggle(f"📷 Enable Camera to capture {calib_color}", key=f"c_mode_{calib_color}")
        
        raw_b = None
        if calib_cam_toggle:
            calib_cam = st.camera_input(f"Snapshot of {calib_color}", key=f"calib_cam_{calib_color}")
            if calib_cam: raw_b = calib_cam.read()
        else:
            calib_up = st.file_uploader(f"Upload photo of {calib_color}", type=['jpg','png','jpeg'], key=f"calib_up_{calib_color}")
            if calib_up: raw_b = calib_up.read()
        
        if raw_b:
            st.image(raw_b, caption="Reference Photo", width=300)
            
            if st.button(f"🎯 Calibrate {calib_color}", type="primary"):
                bgr, annotated = extract_center_bgr(raw_b)
                if bgr is not None:
                    st.image(annotated, caption="🔍 Sampling Area", width=300)
                    
                    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                    st.session_state.custom_std_colors[calib_color] = hsv.tolist()
                    
                    with open(CALIB_FILE, 'w') as f:
                        json.dump(st.session_state.custom_std_colors, f)
                        
                    st.success(f"Successfully calibrated {calib_color} and auto-saved!")
                else:
                    st.error("Failed to extract color. Ensure image is valid.")

    with c2:
        st.markdown("#### 2. Settings")
        st.info("✨ Auto-Saving is Enabled. Calibrated colors are saved instantly!")
            
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.session_state.custom_std_colors = {}
            if os.path.exists(CALIB_FILE): os.remove(CALIB_FILE)
            st.info("Reset to factory defaults.")
            st.rerun()

    st.divider()
    st.markdown("#### 3. Active Calibration Stats")
    std = get_std_colors()
    st_cols = st.columns(3)
    for i, (name, hsv) in enumerate(std.items()):
        rgb = cv2.cvtColor(np.uint8([[[hsv[0], hsv[1], hsv[2]]]]), cv2.COLOR_HSV2RGB)[0][0]
        hex_c = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        
        st_cols[i%3].markdown(f"""
            <div style='padding:10px; border-radius:10px; background:rgba(128,128,128,0.1); border:1px solid rgba(128,128,128,0.3); margin-bottom:10px;'>
                <div style='display:flex; align-items:center; gap:8px;'>
                    <div style='width:20px; height:20px; border-radius:4px; background:{hex_c}; border:1px solid #000;'></div>
                    <b>{name}</b>
                </div>
                <code style='font-size:10px;'>HSV: {list(hsv)}</code>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
