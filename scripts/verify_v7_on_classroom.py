import os
import sys
import cv2
import torch
import json
from ultralytics import YOLO

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

model_path = r"kaggle_results\version7\best.pt"
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} does not exist yet.")
    sys.exit(1)

test_image_path = r"C:\Users\thiet\.gemini\antigravity-cli\brain\30d185aa-58de-4b8e-b83e-50bde963d7f6\.user_uploaded\uploaded_media_1794978411315.jpg"
if not os.path.exists(test_image_path):
    print(f"Error: Image {test_image_path} not found.")
    sys.exit(1)

print(f"Loading YOLO11-M V7 model from {model_path}...")
model = YOLO(model_path)
print(f"Model classes: {model.names}")

img = cv2.imread(test_image_path)
h, w = img.shape[:2]

results = model(test_image_path, conf=0.25, iou=0.45, agnostic_nms=True)[0]

adult_count = 0
child_count = 0
detections = []

annotated_img = img.copy()

for box in results.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    cls_name = model.names[cls_id]
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    if cls_name == 'adult':
        adult_count += 1
        color = (0, 200, 0) # Green for adult
    else:
        child_count += 1
        color = (0, 140, 255) # Orange for child
        
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
    label = f"{cls_name}: {conf:.2f}"
    
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated_img, (x1, max(0, y1 - 25)), (x1 + lw + 6, max(0, y1)), color, -1)
    cv2.putText(annotated_img, label, (x1 + 3, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    detections.append({
        "class": cls_name,
        "confidence": round(conf, 3),
        "bbox": [x1, y1, x2, y2]
    })

status_text = "SAFE: ADULT PRESENT" if adult_count > 0 else "DANGER: UNATTENDED CHILD"
banner_color = (0, 180, 0) if adult_count > 0 else (0, 0, 220)

cv2.rectangle(annotated_img, (0, h - 45), (w, h), banner_color, -1)
banner_msg = f"{status_text} | Adults: {adult_count} | Children: {child_count}"
cv2.putText(annotated_img, banner_msg, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

out_artifact = r"C:\Users\thiet\.gemini\antigravity-cli\brain\30d185aa-58de-4b8e-b83e-50bde963d7f6\classroom_v7_verification.jpg"
cv2.imwrite(out_artifact, annotated_img)

out_local = r"kaggle_results\version7\classroom_v7_verification.jpg"
cv2.imwrite(out_local, annotated_img)

print(f"\nVerification Results:")
print(f"- Adults detected: {adult_count}")
print(f"- Children detected: {child_count}")
print(f"- Status: {status_text}")
print(f"- Saved visual verification to {out_artifact}")
print(json.dumps(detections, indent=2))
