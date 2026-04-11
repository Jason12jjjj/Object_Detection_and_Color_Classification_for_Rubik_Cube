import os
import cv2
import time
import numpy as np

# ==========================================
# 1. Path Configuration (Localized)
# ==========================================
result_folder = "predictions"   # Stores results from svm_predict.py
raw_folder = "to_predict"       # Stores original source images
dataset_path = "svm_dataset"    # Corrected data will be saved here

# ==========================================
# 2. Mouse & Interaction Logic
# ==========================================
# Track mouse click coordinates
click_pos = [None]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = (x, y)

def get_action_from_click(x, y):
    """Determine which virtual button was clicked."""
    if 610 <= y <= 660:
        if 20 <= x <= 100: return 'red'
        if 115 <= x <= 195: return 'green'
        if 210 <= x <= 290: return 'blue'
        if 305 <= x <= 385: return 'yellow'
        if 400 <= x <= 480: return 'orange'
        if 495 <= x <= 575: return 'white'
    elif 680 <= y <= 730:
        if 20 <= x <= 195: return 'skip'
        if 210 <= x <= 385: return 'next'
        if 400 <= x <= 575: return 'quit'
    return None

def draw_control_panel(canvas):
    """Draw UI panel and buttons at the bottom of the window."""
    # Dark grey background footer
    cv2.rectangle(canvas, (0, 600), (600, 750), (40, 40, 40), -1)

    # --- Row 1: Color Selection ---
    # RED
    cv2.rectangle(canvas, (20, 610), (100, 660), (0, 0, 255), -1)
    cv2.putText(canvas, "RED(r)", (35, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # GREEN
    cv2.rectangle(canvas, (115, 610), (195, 660), (0, 200, 0), -1)
    cv2.putText(canvas, "GRN(g)", (125, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # BLUE
    cv2.rectangle(canvas, (210, 610), (290, 660), (255, 0, 0), -1)
    cv2.putText(canvas, "BLU(b)", (225, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # YELLOW
    cv2.rectangle(canvas, (305, 610), (385, 660), (0, 255, 255), -1)
    cv2.putText(canvas, "YLW(y)", (320, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    # ORANGE
    cv2.rectangle(canvas, (400, 610), (480, 660), (0, 140, 255), -1)
    cv2.putText(canvas, "ORG(o)", (415, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    # WHITE
    cv2.rectangle(canvas, (495, 610), (575, 660), (255, 255, 255), -1)
    cv2.putText(canvas, "WHT(w)", (510, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # --- Row 2: Navigation ---
    cv2.rectangle(canvas, (20, 680), (195, 730), (100, 100, 100), -1)
    cv2.putText(canvas, "Skip Cell (Space)", (40, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.rectangle(canvas, (210, 680), (385, 730), (100, 100, 100), -1)
    cv2.putText(canvas, "Next Image (n)", (240, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.rectangle(canvas, (400, 680), (575, 730), (80, 80, 200), -1)
    cv2.putText(canvas, "Quit (q)", (460, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ==========================================
# 3. Main Application Logic
# ==========================================
color_map = {
    ord('r'): 'red', ord('g'): 'green', ord('b'): 'blue',
    ord('y'): 'yellow', ord('o'): 'orange', ord('w'): 'white'
}

def interactive_correct():
    if not os.path.exists(result_folder):
        print(f"Error: Could not find results folder '{result_folder}'")
        return

    result_files = [f for f in os.listdir(result_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not result_files:
        print(f"Status: No images found in '{result_folder}'.")
        return

    print("=== Graphical Data Corrector Started ===")
    print("👉 Use your Mouse to click buttons or Keyboard shortcuts (r/g/b/etc.)")

    # Initialize OpenCV window and bind mouse events
    cv2.namedWindow("Interactive Corrector UI")
    cv2.setMouseCallback("Interactive Corrector UI", mouse_callback, click_pos)

    for res_name in result_files:
        res_path = os.path.join(result_folder, res_name)
        res_img = cv2.imdecode(np.fromfile(res_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if res_img is None: continue
        
        # Determine base name logic
        if "_result" in res_name:
            base_name = res_name.rsplit("_result", 1)[0]
        else:
            base_name = os.path.splitext(res_name)[0]
            
        raw_path = None
        if os.path.exists(raw_folder):
            for file in os.listdir(raw_folder):
                if os.path.splitext(file)[0] == base_name:
                    raw_path = os.path.join(raw_folder, file)
                    break
        
        if raw_path is None or not os.path.exists(raw_path):
            print(f"Warning: Skipping {res_name} (Source image '{base_name}' not found in {raw_folder}).")
            continue
            
        raw_img = cv2.imdecode(np.fromfile(raw_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw_img is None: continue
        
        h, w, _ = raw_img.shape
        gh, gw = h // 3, w // 3
        skip_image = False
        
        for i in range(3):
            if skip_image: break
            for j in range(3):
                # Canvas (600x600 for image, 150 for UI)
                canvas = np.zeros((750, 600, 3), dtype=np.uint8)
                
                # Resize result to 600x600 and highlight current cell
                img_600 = cv2.resize(res_img, (600, 600))
                cv2.rectangle(img_600, (j*200, i*200), ((j+1)*200, (i+1)*200), (0, 0, 255), 6)
                canvas[:600, :] = img_600
                
                # Draw UI labels
                draw_control_panel(canvas)
                
                cv2.imshow("Interactive Corrector UI", canvas)
                click_pos[0] = None # Reset click record
                
                # Event Listening Loop
                action = None
                while True:
                    key = cv2.waitKey(10) & 0xFF
                    # Key checks
                    if key != 255:
                        if key in color_map: action = color_map[key]
                        elif key == ord(' '): action = 'skip'
                        elif key == ord('n'): action = 'next'
                        elif key == ord('q'): action = 'quit'
                    
                    # Mouse checks
                    if click_pos[0] is not None:
                        cx, cy = click_pos[0]
                        action = get_action_from_click(cx, cy)
                        click_pos[0] = None
                    
                    if action: break
                
                # --- Execute Action ---
                if action in ['red', 'green', 'blue', 'yellow', 'orange', 'white']:
                    # Extract pure color patch from original source
                    crop_y, crop_x = int(gh * 0.1), int(gw * 0.1)
                    roi = raw_img[i*gh + crop_y : (i+1)*gh - crop_y, 
                                  j*gw + crop_x : (j+1)*gw - crop_x]
                    
                    target_dir = os.path.join(dataset_path, action)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    save_name = f"fixed_cell_{i}_{j}_{int(time.time()*1000)}.jpg"
                    cv2.imencode('.jpg', roi)[1].tofile(os.path.join(target_dir, save_name))
                    print(f"Action: Cell [{i+1},{j+1}] saved to '{action}' dataset.")
                    
                elif action == 'next': 
                    print("Status: Skipping remaining cells in this image.")
                    skip_image = True
                    break
                elif action == 'quit':
                    print("Status: Exiting...")
                    cv2.destroyAllWindows()
                    return
                elif action == 'skip':
                    continue 

    cv2.destroyAllWindows()
    print("Done: All corrections finished. Run 'svm_train.py' to update the model.")

if __name__ == "__main__":
    interactive_correct()
