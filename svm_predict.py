import os
import cv2
import numpy as np
import joblib
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# Path Configuration (Localized)
# ==========================================
model_path = "svm_color_model.pkl"
to_predict_path = "to_predict"
output_dir = "predictions"

def extract_features_from_block(img_block):
    hsv = cv2.cvtColor(img_block, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def predict_rubiks_face():
    # Ensure directories exist
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(to_predict_path): os.makedirs(to_predict_path)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run svm_train.py first.")
        return

    print("=== Starting 3x3 Matrix Detection (Audit Version) ===")
    svm_model = joblib.load(model_path)
    classes = list(svm_model.classes_)
    
    images_to_predict = [f for f in os.listdir(to_predict_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images_to_predict:
        print(f"Warning: No images found in '{to_predict_path}' folder.")
        return

    for img_name in images_to_predict:
        img_path = os.path.join(to_predict_path, img_name)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        
        file_name = os.path.splitext(img_name)[0]
        h, w, _ = img.shape
        gh, gw = h // 3, w // 3
        
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil, "RGBA") 
        
        try:
            # Attempt to use a system font, fallback to default
            font = ImageFont.truetype("arial.ttf", max(16, int(w/22)))
        except IOError:
            font = ImageFont.load_default()
            
        report_lines = [f"STABLE FULL AUDIT REPORT: {file_name}", "="*65, ""]
        
        for row in range(3):
            for col in range(3):
                y_start, y_end = row * gh, (row + 1) * gh
                x_start, x_end = col * gw, (col + 1) * gw
                
                cell_img = img[y_start:y_end, x_start:x_end]
                # Indent 10% to extract "pure" color area
                crop_y, crop_x = int(gh * 0.1), int(gw * 0.1)
                center_cell = cell_img[crop_y:gh-crop_y, crop_x:gw-crop_x]
                
                feature = extract_features_from_block(center_cell).reshape(1, -1)
                
                # Logic: Use predict_proba to determine color confidence
                probabilities = svm_model.predict_proba(feature)[0]
                max_index = np.argmax(probabilities)
                prediction = classes[max_index]
                max_prob = probabilities[max_index]
                
                # --- Audit TXT Report ---
                report_lines.append(f"▉ GRID [{row+1}, {col+1}] ANALYSIS:")
                report_lines.append(f"  - FINAL DECISION: {prediction} ({max_prob*100:.1f}%)")
                report_lines.append(f"    Status: OK - MATCHES MODEL")
                
                report_lines.append(f"\n  - MODEL PROBABILITY DISTRIBUTION:")
                for k, label in enumerate(classes):
                    p = probabilities[k]
                    bar = "█" * int(p * 20)
                    report_lines.append(f"    {label.ljust(7)}: {p*100:5.2f}% {bar}")
                report_lines.append("-" * 65 + "\n")
                
                # --- UI Labels on Result Image ---
                txt = f"{prediction.upper()} {max_prob*100:.0f}%"
                tx, ty = col * gw + 15, row * gh + 25
                
                try:
                    bbox = draw.textbbox((tx, ty), txt, font=font)
                    draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=(0,0,0,180))
                except AttributeError:
                    draw.rectangle([tx-5, ty-5, tx+100, ty+30], fill=(0,0,0,180))
                draw.text((tx, ty), txt, font=font, fill=(255,255,255))
                
        # Draw Grid Overlay
        final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        for k in range(4): 
            cv2.line(final_img, (0, k*gh), (w, k*gh), (0,255,0), 2)
            cv2.line(final_img, (k*gw, 0), (k*gw, h), (0,255,0), 2)

        # Save Result Image and Audit Report
        res_img_path = os.path.join(output_dir, f"{file_name}_result.jpg")
        cv2.imencode('.jpg', final_img)[1].tofile(res_img_path)
        
        txt_path = os.path.join(output_dir, f"{file_name}_audit.txt")
        with open(txt_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(report_lines))
            
        print(f"Done: {img_name} -> Results saved in '{output_dir}'")

if __name__ == "__main__":
    predict_rubiks_face()
